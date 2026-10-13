"use strict";

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Ошибка запроса (${response.status})`);
  }
  return data;
}

function showError(element, message) {
  if (!element) return;
  element.textContent = message;
  element.hidden = false;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  const inline = (value) =>
    value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const lines = escapeHtml(text || "").split(/\r?\n/);
  const html = [];
  let listType = null;
  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = heading[1].length <= 1 ? 3 : 4;
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const wanted = bullet ? "ul" : "ol";
      if (listType !== wanted) {
        closeList();
        html.push(`<${wanted}>`);
        listType = wanted;
      }
      html.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      continue;
    }
    if (!line.trim()) {
      closeList();
      continue;
    }
    closeList();
    html.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return html.join("");
}

function initAuthForm(formId, endpoint) {
  const form = document.getElementById(formId);
  if (!form) return;
  const errorBox = document.getElementById("error");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    try {
      await api(endpoint, {
        method: "POST",
        body: JSON.stringify({
          username: document.getElementById("username").value,
          password: document.getElementById("password").value,
        }),
      });
      window.location.href = "/chat";
    } catch (error) {
      showError(errorBox, error.message);
    }
  });
}

function initChat() {
  const chatList = document.getElementById("chat-list");
  if (!chatList) return;

  const messagesBox = document.getElementById("messages");
  const titleBox = document.getElementById("chat-title");
  const composer = document.getElementById("composer");
  const input = document.getElementById("message");
  const useRag = document.getElementById("use-rag");
  const state = { chats: [], currentId: null };

  function scrollDown() {
    messagesBox.scrollTop = messagesBox.scrollHeight;
  }

  function renderChats() {
    chatList.innerHTML = "";
    for (const chat of state.chats) {
      const item = document.createElement("li");
      item.className = "chat-item" + (chat.id === state.currentId ? " active" : "");
      item.dataset.id = chat.id;

      const title = document.createElement("span");
      title.className = "title";
      title.textContent = chat.title;
      title.title = chat.title;
      title.addEventListener("click", () => selectChat(chat.id));

      const del = document.createElement("button");
      del.className = "del";
      del.textContent = "✕";
      del.title = "Удалить чат";
      del.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteChat(chat.id);
      });

      item.append(title, del);
      chatList.appendChild(item);
    }
  }

  function sourceList(sources) {
    if (!sources || !sources.length) return "";
    const wrapper = document.createElement("div");
    wrapper.className = "sources";
    wrapper.append("Источники: ");
    sources.forEach((source, index) => {
      const label = source.title || "рецепт";
      if (source.url) {
        const link = document.createElement("a");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = label;
        wrapper.append(link);
      } else {
        wrapper.append(label);
      }
      if (index < sources.length - 1) wrapper.append(", ");
    });
    return wrapper;
  }

  function fillMessage(block, role, content, meta) {
    block.className = `msg ${role}`;
    block.innerHTML = "";
    const mode = meta && meta.mode ? meta.mode : role === "assistant" ? "rag" : null;
    if (role === "assistant" && mode) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = mode === "rag" ? "рецепты" : "без RAG";
      block.appendChild(badge);
    }
    const text = document.createElement("div");
    text.className = "md";
    text.innerHTML = renderMarkdown(content);
    block.appendChild(text);
    const sources = sourceList(meta && meta.sources);
    if (sources) block.appendChild(sources);
  }

  function renderMessage(role, content, meta) {
    const block = document.createElement("div");
    fillMessage(block, role, content, meta);
    messagesBox.appendChild(block);
    return block;
  }

  function renderThinking() {
    const block = document.createElement("div");
    block.className = "msg assistant thinking";
    const label = document.createElement("span");
    label.textContent = "Модель думает";
    const dots = document.createElement("span");
    dots.className = "dots";
    for (let i = 0; i < 3; i += 1) dots.appendChild(document.createElement("i"));
    block.append(label, dots);
    messagesBox.appendChild(block);
    return block;
  }

  function clearMessages(placeholder) {
    messagesBox.innerHTML = "";
    if (placeholder) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = placeholder;
      messagesBox.appendChild(empty);
    }
  }

  async function loadStatus() {
    try {
      const data = await api("/api/status");
      const statusBox = document.getElementById("status");
      statusBox.textContent =
        `${data.model} · ${data.dataset}/${data.strategy} · ` +
        `${data.total_vectors} чанков` + (data.offline ? " · офлайн" : "");
    } catch (error) {
      document.getElementById("status").textContent = error.message;
    }
  }

  async function loadChats() {
    const data = await api("/api/chats");
    state.chats = data.chats;
    if (!state.currentId && state.chats.length) {
      await selectChat(state.chats[0].id);
    } else {
      renderChats();
    }
  }

  async function selectChat(id) {
    state.currentId = id;
    const chat = state.chats.find((item) => item.id === id);
    titleBox.textContent = chat ? chat.title : "Чат";
    renderChats();
    const data = await api(`/api/chats/${id}/messages`);
    clearMessages("История пуста. Задайте вопрос по рецептам.");
    for (const item of data.messages) {
      renderMessage(item.role, item.content, item.meta);
    }
    scrollDown();
  }

  async function createChat() {
    const data = await api("/api/chats", { method: "POST", body: JSON.stringify({}) });
    state.chats.unshift(data.chat);
    await selectChat(data.chat.id);
  }

  async function deleteChat(id) {
    await api(`/api/chats/${id}`, { method: "DELETE" });
    state.chats = state.chats.filter((chat) => chat.id !== id);
    if (state.currentId === id) {
      state.currentId = null;
      titleBox.textContent = "Выберите чат";
      clearMessages("Создайте чат и задайте вопрос по базе рецептов.");
    }
    renderChats();
  }

  async function resetChat() {
    if (!state.currentId) return;
    await api(`/api/chats/${state.currentId}/reset`, { method: "POST" });
    clearMessages("История пуста. Задайте вопрос по рецептам.");
  }

  async function send(event) {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || !state.currentId) return;
    renderMessage("user", text, null);
    input.value = "";
    scrollDown();
    const submit = composer.querySelector("button");
    submit.disabled = true;
    const pending = renderThinking();
    scrollDown();
    try {
      const result = await api(`/api/chats/${state.currentId}/messages`, {
        method: "POST",
        body: JSON.stringify({ message: text, use_rag: useRag.checked }),
      });
      fillMessage(pending, "assistant", result.reply, {
        mode: result.mode,
        sources: result.sources,
      });
      const chat = state.chats.find((item) => item.id === state.currentId);
      if (chat && chat.title === "Новый чат") {
        await loadChats();
        const updated = state.chats.find((item) => item.id === state.currentId);
        if (updated) titleBox.textContent = updated.title;
      }
    } catch (error) {
      fillMessage(pending, "assistant", `⚠️ ${error.message}`, { mode: "plain" });
    } finally {
      submit.disabled = false;
      scrollDown();
    }
  }

  document.getElementById("new-chat").addEventListener("click", createChat);
  document.getElementById("reset-chat").addEventListener("click", resetChat);
  document.getElementById("logout").addEventListener("click", async () => {
    await api("/api/logout", { method: "POST" });
    window.location.href = "/login";
  });
  composer.addEventListener("submit", send);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  loadStatus();
  loadChats().catch((error) => showError(document.getElementById("status"), error.message));
}

document.addEventListener("DOMContentLoaded", () => {
  initAuthForm("login-form", "/api/login");
  initAuthForm("register-form", "/api/register");
  initChat();
});
