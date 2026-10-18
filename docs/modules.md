# Справочник по модулям

> Файл сгенерирован автоматически командой `python3 scripts/generate_docs.py`.
> Не редактируйте его вручную — изменения будут перезаписаны.


Всего модулей: **72**.

## Пакет `agent/`

### `agent/__init__.py` (2 строк)

Пакет агента: отдельная сущность, вызывающая LLM через API.

### `agent/agent.py` (572 строк)

Агент — отдельная сущность, принимающая запрос и вызывающая LLM.

**Классы:**

- `Agent` — Инкапсулирует логику запроса к LLM и возврата ответа.
  - `context_strategy(self) -> ContextStrategy | None` — Стратегия управления контекстом (None — история как есть).
  - `conversation(self) -> Conversation`
  - `history(self) -> list[dict]` — История диалога текущей сессии.
  - `usage(self) -> UsageReport` — Накопительная статистика токенов и стоимости сессии.
  - `compression(self) -> CompressionConfig` — Настройки сжатия истории.
  - `memory(self) -> MemoryLayers | None` — Явная модель памяти агента (None — память не используется).
  - `personalization(self) -> Personalization | None` — Персонализация агента: активный профиль пользователя (None — выключена).
  - `task(self) -> TaskStateMachine | None` — Конечный автомат состояния задачи (None — состояние не используется).
  - `invariants(self) -> Invariants | None` — Инварианты агента (None — ограничения не используются).
  - `last_guard_violation(self) -> bool` — Нарушал ли последний ответ агента инварианты (True — был отказ).
  - `last_transition_violation(self) -> bool` — Перепрыгивал ли последний ответ агента этап задачи (True — был отказ).
  - `tools(self) -> McpToolRuntime | None` — Рантайм MCP-инструментов (None — инструменты не подключены).
  - `last_tool_call(self) -> ToolCall | None` — Какой MCP-инструмент вызвал агент на последнем ходу (None — не вызывал).
  - `last_tool_result(self) -> ToolResult | None` — Результат последнего вызова MCP-инструмента (None — вызова не было).
  - `max_tool_steps(self) -> int` — Максимум шагов длинного флоу инструментов за один ход.
  - `last_tool_calls(self) -> list[ToolCall]` — Все вызовы MCP-инструментов за последний ход, в порядке выполнения.
  - `last_tool_results(self) -> list[ToolResult]` — Результаты всех вызовов инструментов за последний ход, по порядку.
  - `save_task(self) -> None` — Сохраняет текущее состояние задачи в сессию (чтобы оно пережило перезапуск).
  - `profile_prompt(self) -> str` — System-сообщение активного профиля (пусто, если профиля нет/пуст).
  - `context_messages(self) -> list[dict]` — Сообщения, которые будут отправлены в модель.
  - `context_tokens(self) -> int` — Токены текущего контекста, отправляемого в модель.
  - `ask(self, user_request: str, temperature: float | None = None, max_tokens: int | None = None, context_window: int | None = None) -> Completion` — Отправляет запрос в модель.

### `agent/cli.py` (983 строк)

CLI-интерфейс: принимает запрос пользователя и выводит ответ агента.

**Функции:**

- `build_agent(conversation) -> Agent`
- `print_usage(agent: Agent) -> None` — Печатает токены и стоимость последнего хода и всей сессии.
- `print_history(agent: Agent) -> None` — Выводит полную историю диалога, помечая сообщения, свёрнутые в summary.
- `pick_session(store: SessionStore) -> Agent` — Показывает список сессий и возвращает агента для выбранной или новой.
- `print_memory_status(agent: Agent) -> None` — Печатает сводку по всем слоям памяти агента.
- `print_task_status(agent: Agent) -> None` — Печатает формализованное состояние текущей задачи (конечный автомат).
- `print_invariant_status(agent: Agent) -> None` — Печатает все установленные инварианты агента.
- `print_context_status(agent: Agent) -> None` — Печатает активную стратегию управления контекстом и состояние веток.
- `print_help() -> None` — Печатает список доступных команд агента.
- `main() -> None`

### `agent/code_index.py` (446 строк)

День 32: индексация исходного кода проекта для RAG-ревью.

**Классы:**

- `CodeSymbol` — Один top-level символ модуля (класс, функция, константы, шапка).
- `CodeModule` — Python-модуль, попадающий в RAG-индекс кода.

**Функции:**

- `discover_code_files(root: Path = PROJECT_ROOT, globs: tuple[str, ...] = DEFAULT_CODE_GLOBS) -> list[Path]` — Находит Python-файлы проекта по glob-шаблонам (отсортировано).
- `parse_module_symbols(text: str, max_symbol_chars: int = DEFAULT_MAX_SYMBOL_CHARS) -> list[CodeSymbol]` — Извлекает top-level символы модуля через ``ast``.
- `chunk_code_module(module: CodeModule, max_symbol_chars: int = DEFAULT_MAX_SYMBOL_CHARS) -> list[Chunk]` — Превращает модуль в чанки с метаданными о файле и символе.
- `load_code_module(path: Path, root: Path = PROJECT_ROOT) -> CodeModule` — Загружает один Python-модуль и разбирает его символы.
- `load_code_modules(root: Path = PROJECT_ROOT, globs: tuple[str, ...] = DEFAULT_CODE_GLOBS) -> list[CodeModule]` — Загружает все исходники проекта.
- `build_code_index(root: Path = PROJECT_ROOT, index_dir: str | Path = DEFAULT_CODE_INDEX_DIR, dataset: str = DEFAULT_CODE_DATASET, strategy: str = DEFAULT_CODE_STRATEGY, max_symbol_chars: int = DEFAULT_MAX_SYMBOL_CHARS, embedder: HashingEmbedder | None = None, modules: list[CodeModule] | None = None, save: bool = True) -> tuple[FaissIndex, list[CodeModule]]` — Собирает (и при ``save=True`` сохраняет) индекс исходного кода проекта.
- `load_code_retriever(index_dir: str | Path = DEFAULT_CODE_INDEX_DIR, dataset: str = DEFAULT_CODE_DATASET, strategy: str = DEFAULT_CODE_STRATEGY, top_k: int = 5, min_score: float = 0.0, root: Path = PROJECT_ROOT) -> RagRetriever` — Загружает индекс кода с диска как :class:`RagRetriever`.
- `build_code_context(sources: list[RagSource], max_chars: int = 6000) -> str` — Собирает контекст кода с указанием файла и символа.
- `main() -> int`

### `agent/code_review.py` (1232 строк)

День 32: ассистент автоматического ревью кода.

**Классы:**

- `ReviewFinding` — Одна находка ревью.
  - `to_dict(self) -> dict[str, Any]`
- `CodeReview` — Результат автоматического ревью pull request.
  - `by_category(self, category: str) -> list[ReviewFinding]`
  - `counts(self) -> dict[str, int]`
  - `max_severity(self) -> str | None`
  - `to_dict(self) -> dict[str, Any]`
  - `to_markdown(self) -> str` — Человекочитаемое ревью (для комментария к PR и артефакта).
- `ReviewStubClient` — Детерминированный офлайн-клиент ревью.
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`
- `CodeReviewAssistant` — Ассистент ревью: PR-контекст (MCP) + RAG (документация + код) + LLM.
  - `total_vectors(self) -> int`
  - `collect_pr_context(self, base: str, head: str = '') -> dict[str, Any]` — Собирает контекст PR: diff, изменённые файлы, метаданные.
  - `retrieve_context(self, changed_files: list[dict], diff: str, top_k: int | None = None) -> tuple[list[RagSource], list[RagSource]]` — Извлекает релевантную документацию и код по изменённым файлам.
  - `review(self, base: str = 'HEAD', head: str = '', pr: dict | None = None) -> CodeReview` — Проводит ревью PR и возвращает структурированный результат.

**Функции:**

- `parse_added_lines(diff: str) -> list[_AddedLine]` — Разбирает unified diff и возвращает только добавленные строки.
- `analyze_diff(diff: str) -> list[ReviewFinding]` — Детерминированные эвристики по добавленным строкам diff.
- `parse_findings(answer_text: str) -> tuple[str, list[ReviewFinding]]` — Разбирает ответ модели на резюме и список находок.
- `render_pr_metadata(pr: dict[str, Any] | None) -> str` — Человекочитаемые метаданные PR для промпта.
- `render_changed_files(changed: list[dict]) -> str` — Список изменённых файлов для промпта.
- `build_review_messages(pr: dict[str, Any], changed_files: list[dict], diff: str, docs_sources: list[RagSource], code_sources: list[RagSource], max_chars: int = 8000, max_diff_chars: int = 20000, system_prompt: str | None = None) -> list[dict[str, str]]` — Формирует сообщения для ассистента ревью: PR + RAG (доки и код).
- `build_review_config(config: dict | None = None, env: dict | None = None) -> dict` — Собирает конфиг LLM-клиента, подмешивая переменные окружения.
- `has_llm_credentials(config: dict) -> bool` — Есть ли в конфиге всё для обращения к LLM API.
- `build_review_runtime(config: dict | None = None) -> McpToolRuntime | None` — Создаёт MCP-рантайм сервера ревью из конфигурации.
- `build_review_assistant(config: dict | None = None, offline: bool = False, docs_index_dir: str = DEFAULT_DOCS_INDEX_DIR, docs_dataset: str = DEFAULT_DOCS_DATASET, docs_strategy: str = DEFAULT_DOCS_STRATEGY, code_index_dir: str = DEFAULT_CODE_INDEX_DIR, code_dataset: str = DEFAULT_CODE_DATASET, code_strategy: str = DEFAULT_CODE_STRATEGY, top_k: int = 5, min_score: float = 0.0, temperature: float = 0.0, max_context_chars: int = 8000, max_diff_chars: int = 20000, build_index_if_missing: bool = True, with_review_mcp: bool = True) -> CodeReviewAssistant` — Собирает ассистента ревью: оба индекса, MCP-рантайм, LLM-клиент.

### `agent/compression.py` (142 строк)

Сжатие истории диалога: старые сообщения заменяются кратким summary.

**Классы:**

- `CompressionConfig` — Настройки сжатия истории.
  - `enabled(self) -> bool` — Сжатие включено, если окно свежих сообщений ограничено.

**Функции:**

- `build_context_messages(conversation: Conversation) -> list[dict]` — Собирает сообщения для отправки в модель.
- `build_summary_prompt(messages: list[dict], old_summary: str = '') -> list[dict]` — Формирует сообщения для запроса на создание/обновление summary.
- `summarize_messages(client, messages: list[dict], old_summary: str = '', max_tokens: int | None = None) -> str` — Просит LLM сжать список сообщений в краткое содержание.
- `should_compress(conversation: Conversation, config: CompressionConfig) -> bool` — Пора ли сжимать: свежих сообщений накопилось заметно больше keep_recent.
- `compress(conversation: Conversation, client, config: CompressionConfig) -> bool` — Сжимает историю: свежие сообщения сворачиваются в обновлённый summary.
- `compressed_context_tokens(conversation: Conversation) -> int` — Оценка токенов сжатого контекста: summary + свежие сообщения.

### `agent/config.py` (12 строк)

Загрузка и доступ к конфигурации агента.

**Функции:**

- `load_config(path: Path = CONFIG_PATH) -> dict`

### `agent/context.py` (233 строк)

Стратегии управления контекстом диалога агента.

**Классы:**

- `ContextStrategy` (ABC) — Общий интерфейс стратегии управления контекстом.
  - `name(self) -> str` — Короткое имя стратегии (для логов и конфигурации).
  - `label(self) -> str` — Человекочитаемое название стратегии.
  - `build_messages(self, conversation: Conversation) -> list[dict]` — Возвращает сообщения, которые будут отправлены в модель.
  - `context_tokens(self, conversation: Conversation) -> int` — Оценка токенов контекста, отправляемого в модель.
  - `on_user_message(self, conversation: Conversation, user_request: str) -> None` — Хук после добавления сообщения пользователя (напр. извлечение фактов).
  - `on_reply(self, conversation: Conversation, reply) -> None` — Хук после добавления ответа ассистента (напр. усечение окна).
- `SlidingWindowStrategy` (ContextStrategy) — Стратегия 1: храним только последние ``size`` сообщений.
  - `name(self) -> str`
  - `label(self) -> str`
  - `build_messages(self, conversation: Conversation) -> list[dict]`
  - `context_tokens(self, conversation: Conversation) -> int`
  - `on_reply(self, conversation: Conversation, reply) -> None`
- `StickyFactsStrategy` (ContextStrategy) — Стратегия 2: липкие факты (ключ-значение) + последние ``window_size`` сообщений.
  - `name(self) -> str`
  - `label(self) -> str`
  - `build_messages(self, conversation: Conversation) -> list[dict]`
  - `context_tokens(self, conversation: Conversation) -> int`
  - `on_user_message(self, conversation: Conversation, user_request: str) -> None`
- `BranchingStrategy` (ContextStrategy) — Стратегия 3: ветвление диалога от checkpoint.
  - `name(self) -> str`
  - `label(self) -> str`
  - `build_messages(self, conversation: Conversation) -> list[dict]`
  - `context_tokens(self, conversation: Conversation) -> int`

**Функции:**

- `extract_facts_from_text(text: str) -> dict[str, str]` — Эвристически извлекает пары ключ-значение из реплики пользователя.
- `strategy_from_config(config: dict) -> ContextStrategy | None` — Создаёт стратегию управления контекстом из секции ``context`` конфигурации.

### `agent/context_scenarios.py` (229 строк)

Day 10: стратегии управления контекстом — живое сравнение без хардкода.

**Классы:**

- `FakeClient` — Имитирует LLM: считает токены локально, не отправляя запросы в сеть.
  - `complete(self, messages, temperature = None, max_tokens = None) -> Completion`
- `Runner` — Держит по одному агенту на каждую стратегию и прогоняет общий ввод.
  - `agents(self) -> dict[str, Agent]`
  - `handle_command(self, command: str) -> bool` — Обрабатывает служебную команду. Возвращает True, если это команда.
  - `ask_all(self, user_request: str) -> None` — Отправляет реплику всем агентам и печатает сравнение.
  - `print_status(self) -> None` — Сводка по всем стратегиям: контекст, токены, стоимость.

**Функции:**

- `build_client(config: dict, offline: bool) -> object`
- `new_agent(config: dict, offline: bool, strategy: ContextStrategy | None) -> Agent` — Создаёт агента для одной стратегии в отдельной временной сессии.
- `main() -> None`

### `agent/conversation.py` (363 строк)

Сессии диалога агента: история каждой беседы сохраняется в отдельный JSON.

**Классы:**

- `Conversation` — История одной сессии диалога, хранящаяся в JSON-файле.
  - `session_id(self) -> str`
  - `messages(self) -> list[dict]` — Активный диалог: общая часть (checkpoint) + сообщения активной ветки.
  - `summary(self) -> str` — Краткое содержание старой части диалога (пусто, если сжатия не было).
  - `summarized(self) -> int` — Сколько первых сообщений истории уже свёрнуто в summary.
  - `recent_messages(self) -> list[dict]` — Сообщения, которые ещё не свёрнуты в summary (свежее окно).
  - `facts(self) -> dict[str, str]` — Блок "липких фактов" (ключ-значение), накопленных за диалог.
  - `working(self) -> dict[str, str]` — Рабочая память: данные текущей задачи (ключ-значение).
  - `task(self) -> dict` — Состояние задачи (конечный автомат), сериализованное в словарь.
  - `active_branch(self) -> str | None` — Имя активной ветки диалога (None, если ветвление не начато).
  - `branches(self) -> dict[str, dict]` — Все ветки диалога: id -> {name, messages} (без общих сообщений).
  - `has_branches(self) -> bool` — Есть ли в диалоге созданные ветки.
  - `load(self) -> None` — Загружает историю из JSON-файла сессии.
  - `save(self) -> None` — Сохраняет историю в JSON-файл сессии.
  - `append(self, role: str, content: str, meta: dict | None = None) -> None` — Добавляет сообщение в активную ветку (или общую часть) и сохраняет.
  - `set_summary(self, summary: str, summarized: int | None = None) -> None` — Устанавливает краткое содержание старой части диалога.
  - `pop_last(self) -> None` — Удаляет последнее сообщение активной ветки и сохраняет.
  - `clear(self) -> None` — Очищает историю, summary, факты и ветки сессии.
  - `trim_to(self, keep: int) -> None` — Жёстко отбрасывает всё, кроме последних ``keep`` сообщений (без summary).
  - `set_fact(self, key: str, value: str) -> None` — Записывает один липкий факт и сохраняет.
  - `update_facts(self, facts: dict[str, str]) -> None` — Добавляет/обновляет несколько липких фактов и сохраняет.
  - `set_working(self, key: str, value: str) -> None` — Записывает одну запись в рабочую память (данные текущей задачи).
  - `update_working(self, items: dict[str, str]) -> None` — Добавляет/обновляет несколько записей рабочей памяти.
  - `clear_working(self) -> None` — Очищает рабочую память (например, при смене текущей задачи).
  - `set_task(self, task: dict) -> None` — Сохраняет состояние задачи (конечный автомат) в сессию.
  - `clear_task(self) -> None` — Сбрасывает состояние задачи (новая задача).
  - `checkpoint(self) -> int` — Сохраняет текущее состояние как общую точку ветвления.
  - `create_branch(self, name: str) -> bool` — Создаёт независимую ветку от точки ветвления и переключается на неё.
  - `switch_branch(self, name: str) -> bool` — Переключает активную ветку. Возвращает False, если такой ветки нет.
- `SessionStore` — Управляет списком доступных сессий диалога.
  - `list_sessions(self) -> list[str]` — Возвращает идентификаторы сессий, отсортированные от старых к новым.
  - `create(self) -> Conversation` — Создаёт и возвращает новую пустую сессию.
  - `get(self, session_id: str) -> Conversation` — Открывает существующую сессию.

### `agent/dev_assistant.py` (565 строк)

День 31: ассистент разработчика — RAG по документации + контекст проекта через MCP.

**Классы:**

- `DevAssistantStubClient` — Детерминированный офлайн-клиент ассистента разработчика.
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`
- `DevAnswer` — Ответ ассистента разработчика.
  - `to_dict(self) -> dict[str, Any]`
- `DevAssistant` — Ассистент разработчика: RAG по документации + контекст проекта через MCP.
  - `total_vectors(self) -> int`
  - `retrieve(self, question: str, top_k: int | None = None) -> list[RagSource]`
  - `collect_project_context(self, question: str) -> dict[str, Any]` — Собирает живой контекст проекта через MCP-инструменты.
  - `ask(self, question: str, top_k: int | None = None, history: list[dict[str, str]] | None = None) -> DevAnswer` — Отвечает на вопрос о проекте по документации и контексту.
  - `answer_help(self, question: str, history: list[dict[str, str]] | None = None) -> DevAnswer` — Ответ на команду ``/help`` — вопрос о проекте.
  - `get_branch(self) -> dict[str, Any]` — Текущая git-ветка (для команды ``/branch``).
  - `list_files(self, path: str = '', limit: int = 100) -> dict[str, Any]`
  - `get_diff(self, path: str = '', max_chars: int = 4000) -> dict[str, Any]`

**Функции:**

- `render_project_context(context: dict[str, Any]) -> str` — Человекочитаемое представление контекста проекта для модели.
- `build_dev_messages(question: str, sources: list[RagSource], project_context: dict[str, Any], max_chars: int = 6000, history: list[dict[str, str]] | None = None, system_prompt: str | None = None) -> list[dict[str, str]]` — Формирует сообщения для ассистента: документация + контекст проекта.
- `build_project_runtime(config: dict | None = None) -> McpToolRuntime | None` — Создаёт MCP-рантайм сервера проекта из конфигурации.
- `build_dev_assistant(config: dict | None = None, offline: bool = False, index_dir: str = DEFAULT_INDEX_DIR, dataset: str = DEFAULT_DATASET, strategy: str = DEFAULT_STRATEGY, top_k: int = 5, min_score: float = 0.0, temperature: float = 0.2, max_context_chars: int = 6000, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP, build_index_if_missing: bool = True, with_project_mcp: bool = True) -> DevAssistant` — Собирает ассистента разработчика: индекс, MCP-рантайм, LLM-клиент.

### `agent/dev_assistant_cli.py` (272 строк)

День 31: интерактивный CLI ассистента разработчика.

**Функции:**

- `print_banner(assistant: DevAssistant) -> None`
- `handle_command(raw: str, assistant: DevAssistant, state: dict) -> bool` — Обрабатывает slash-команду. True — ввод был командой.
- `main() -> int`

### `agent/dev_assistant_scenarios.py` (331 строк)

День 31: сквозной сценарий ассистента разработчика (офлайн-проверка).

**Функции:**

- `now_iso() -> str`
- `check_index(index_dir: str, dataset: str, strategy: str) -> dict` — Собирает индекс документации и проверяет состав источников.
- `check_mcp(runtime, limit: int = 10) -> dict` — Проверяет MCP-инструменты проекта (минимум — get_current_branch).
- `check_help(assistant: DevAssistant, questions: list[dict]) -> list[dict]` — Проверяет ответы ``/help`` на вопросы о структуре проекта.
- `render_report(index_info: dict, mcp_info: dict, help_results: list[dict], coverage_ok: bool) -> str`
- `main() -> int`

### `agent/file_assistant.py` (776 строк)

День 34: ассистент для работы с файлами проекта.

**Классы:**

- `FileAction` — Одно действие ассистента над файлами (вызов MCP-инструмента).
  - `to_dict(self) -> dict[str, Any]`
- `FileFinding` — Находка: использование, нарушение правила и т.п.
  - `to_dict(self) -> dict[str, Any]`
- `FileTaskResult` — Результат выполнения цели ассистентом.
  - `to_dict(self) -> dict[str, Any]`
- `FileAssistantStubClient` — Детерминированный офлайн-генератор содержимого документов.
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`
- `FileAssistant` — Ассистент файловых операций: цель → MCP-инструменты → результат.
  - `read_file(self, path: str, max_chars: int | None = None) -> dict[str, Any]`
  - `search_files(self, query: str, path: str = '', glob: str = '*.py', limit: int = 100, context: int = 0) -> dict[str, Any]`
  - `analyze_file(self, path: str) -> dict[str, Any]`
  - `list_files(self, path: str = '', pattern: str = '', limit: int = 200) -> dict[str, Any]`
  - `get_diff(self, path: str = '', baseline_root: str = '', max_chars: int = 8000) -> dict[str, Any]`
  - `write_file(self, path: str, content: str, overwrite: bool = True) -> dict[str, Any]`
  - `run(self, goal: str, *, task_type: str | None = None, apply: bool | None = None, symbol: str = '', kind: str = '', target: str = '', doc_path: str = '', source_paths: list[str] | None = None, baseline_root: str = '', path: str = '') -> FileTaskResult` — Выполняет цель ассистента, самостоятельно выбирая файлы и инструменты.

**Функции:**

- `classify_goal(goal: str) -> str` — Определяет тип задачи по формулировке цели.
- `extract_symbol(goal: str) -> str` — Извлекает искомый символ из цели (в бэктиках/кавычках или эвристикой).
- `render_document(kind: str, facts: dict[str, Any]) -> str` — Собирает markdown-документ из структурированных фактов (без LLM).
- `build_fs_runtime(config: dict | None = None, workspace_root: str | Path | None = None) -> McpToolRuntime | None` — Создаёт MCP-рантайм сервера файлов из конфигурации.
- `build_file_assistant(config: dict | None = None, offline: bool = False, workspace_root: str | Path = PROJECT_ROOT, apply: bool = False, max_chars: int = 8000, temperature: float = 0.0, with_fs_mcp: bool = True) -> FileAssistant` — Собирает ассистента файловых операций: MCP-рантайм + клиент.

### `agent/file_cli.py` (279 строк)

День 34: интерактивный CLI ассистента для работы с файлами.

**Функции:**

- `print_banner(assistant: FileAssistant) -> None`
- `handle_command(raw: str, assistant: FileAssistant, state: dict) -> bool` — Обрабатывает slash-команду. True — ввод был командой.
- `main() -> int`

### `agent/file_scenarios.py` (359 строк)

День 34: сквозной сценарий ассистента файловых операций (офлайн-проверка).

**Функции:**

- `now_iso() -> str`
- `prepare_sandbox(work_dir: Path) -> dict[str, Path]` — Создаёт (пересоздаёт) песочницу и возвращает её пути.
- `check_find_usages(symbol: str, config: dict) -> dict` — Сценарий 1: поиск использований символа по многим файлам (чтение).
- `check_generate(artifacts: Path, config: dict) -> dict` — Сценарий 2: генерация нового файла из исходников (запись + diff).
- `check_invariants(invariants_dir: Path, config: dict) -> dict` — Сценарий 3: проверка соответствия файлов правилам (инварианты).
- `check_update_docs(baseline: Path, work: Path, config: dict) -> dict` — Сценарий 4: обновление документации по изменениям в коде + diff.
- `render_report(scenarios: dict, all_ok: bool) -> str`
- `main() -> int`

### `agent/grounding.py` (1401 строк)

Day 24: Цитаты, источники и анти-галлюцинации (Grounded RAG).

**Классы:**

- `GroundedSource` — Источник, на который ссылается модель в ответе (source + section/chunk_id).
  - `to_dict(self) -> dict[str, Any]`
  - `from_rag_source(cls, src: RagSource) -> GroundedSource` — Создает GroundedSource из чанка поисковой выдачи.
- `AlignmentEvaluation` — Результаты проверки смыслового соответствия ответа цитатам.
  - `to_dict(self) -> dict[str, Any]`
- `GroundedAnswer` — Результат выполнения заземлённого RAG-запроса.
  - `to_dict(self) -> dict[str, Any]`
  - `format_display(self) -> str` — Форматирует красивый текстовый вывод для терминала.
- `GroundedRagAgent` — Интерактивный кулинарный агент с гарантированным заземлением, источниками и цитатами.
  - `ask(self, question: str, expected_facts: list[list[str]] | None = None) -> GroundedAnswer` — Отправляет вопрос ассистенту с заземлением на цитаты и источники.

**Функции:**

- `build_grounded_context(sources: list[RagSource], max_chars: int = 6000) -> str` — Собирает обогащённый контекст с четкими метаданными: source, section, chunk_id.
- `build_grounded_messages(question: str, sources: list[RagSource], max_chars: int = 6000) -> list[dict[str, str]]` — Формирует список сообщений (messages) для заземлённого RAG-запроса.
- `check_relevance_guardrail(sources: list[RagSource], threshold: float = 0.33) -> tuple[bool, float]` — Проверяет порог релевантности найденных чанков.
- `parse_grounded_response(raw_content: str, retrieved_sources: list[RagSource]) -> tuple[str, list[GroundedSource], list[str], bool, bool]` — Разбирает ответ модели, извлекая answer, sources и quotes.
- `verify_sources_presence(answer: GroundedAnswer) -> dict[str, Any]` — Проверяет: есть ли источники в ответе (source + section/chunk_id).
- `verify_quotes_presence(answer: GroundedAnswer) -> dict[str, Any]` — Проверяет: есть ли цитаты в ответе (непустые содержательные строки).
- `verify_quote_verbatim_faithfulness(quotes: list[str], retrieved_sources: list[RagSource]) -> dict[str, Any]` — Анти-галлюцинация: проверяет, что цитаты реально существуют в найденных чанках.
- `verify_semantic_alignment(answer_text: str, quotes: list[str], retrieved_sources: list[RagSource] | None = None, expected_facts: list[list[str]] | None = None, query: str = '', sources: list[GroundedSource] | None = None) -> AlignmentEvaluation` — Проверяет: совпадает ли смысл ответа с цитатами (Semantic Alignment).
- `verify_refusal_and_clarification(text: str) -> dict[str, Any]` — Проверяет правило усиления: 'не знаю' + запрос уточнения.
- `grounded_rag_query(client: LLMClient | None, retriever: RagRetriever, question: str, top_k: int = 5, relevance_threshold: float = 0.33, temperature: float = 0.2, max_context_chars: int = 6000, expected_facts: list[list[str]] | None = None, rerank_pipeline: RerankPipeline | None = None) -> GroundedAnswer` — Полный пайплайн заземлённого RAG-запроса с цитатами, источниками и анти-галлюцинациями.
- `evaluate_grounded_benchmark(agent: GroundedRagAgent, questions: list[dict[str, Any]] | None = None, progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]` — Запускает полную проверку 10 вопросов на источники, цитаты и смысловое соответствие.
- `evaluate_low_relevance_guardrail(agent: GroundedRagAgent, questions: list[dict[str, Any]] | None = None, progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]` — Проверяет правило усиления на вопросах с релевантностью ниже порога.
- `render_grounding_markdown_report(benchmark_summary: dict[str, Any], guardrail_summary: dict[str, Any] | None = None, relevance_threshold: float = 0.33) -> str` — Генерирует подробный Markdown-отчёт о проверке источников, цитат, смыслового соответствия и порога релевантности.

### `agent/grounding_scenarios.py` (308 строк)

Day 24: Сквозной сценарий цитирования, источников и анти-галлюцинаций (Grounded RAG).

**Функции:**

- `main() -> None`

### `agent/indexing.py` (1062 строк)

Day 21: Пайплайн индексации документов (рецептов).

**Классы:**

- `Chunk` — Единица индексации: текст и сопутствующие структурированные метаданные.
  - `to_dict(self) -> dict[str, Any]`
  - `from_dict(cls, data: dict[str, Any]) -> Chunk`
- `Embedder` (Protocol) — Интерфейс генератора эмбеддингов.
  - `embed(self, texts: list[str]) -> np.ndarray` — Возвращает массив векторов размера (len(texts), dim) с L2-нормой 1.
  - `embed_one(self, text: str) -> np.ndarray` — Возвращает один вектор размера (dim,) с L2-нормой 1.
- `HashingEmbedder` — Детерминированный офлайн-эмбеддер на основе хеширования признаков.
  - `embed_one(self, text: str) -> np.ndarray`
  - `embed(self, texts: list[str]) -> np.ndarray`
- `OpenAIEmbedder` — Эмбеддер через OpenAI-совместимый API (/embeddings).
  - `embed(self, texts: list[str]) -> np.ndarray`
  - `embed_one(self, text: str) -> np.ndarray`
- `FaissIndex` — Локальный векторный индекс FAISS (IndexFlatIP) с сохранением метаданных.
  - `total(self) -> int`
  - `build(cls, chunks: list[Chunk], embedder: Embedder, metadata: dict[str, Any] | None = None) -> FaissIndex` — Строит FAISS-индекс по списку чанков.
  - `save(self, directory: str | Path) -> Path` — Сохраняет FAISS-индекс и сопутствующие файлы в директорию:
  - `load(cls, directory: str | Path, embedder: Embedder) -> FaissIndex` — Загружает индекс и метаданные из директории.
  - `search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]` — Ищет ближайшие чанки по запросу.

**Функции:**

- `load_recipes(source: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]` — Загружает рецепты из JSON-файла.
- `recipe_to_text(recipe: dict[str, Any]) -> str` — Формирует полный канонический текст рецепта для сплошного чанкинга.
- `chunk_fixed_size(recipes: list[dict[str, Any]], chunk_size: int = 400, chunk_overlap: int = 80, source: str = '') -> list[Chunk]` — Стратегия 1: разбиение по фиксированному размеру окна со сдвигом.
- `chunk_structural(recipes: list[dict[str, Any]], source: str = '') -> list[Chunk]` — Стратегия 2: разбиение по логической структуре рецепта.
- `compute_chunking_stats(chunks: list[Chunk]) -> dict[str, Any]` — Вычисляет статистику распределения размеров и разделов чанков.
- `evaluate_retrieval(index: FaissIndex, queries: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]` — Оценивает качество поиска по тестовым запросам.
- `compare_strategies(recipes: list[dict[str, Any]], source_name: str = 'recipes', chunk_size: int = 400, chunk_overlap: int = 80, embedder: Embedder | None = None, top_k: int = 5, queries: list[dict[str, Any]] | None = None) -> dict[str, Any]` — Сравнивает две стратегии chunking (fixed vs structural).
- `render_comparison_markdown(report: dict[str, Any]) -> str` — Формирует понятный Markdown-отчёт со сравнением двух стратегий.

### `agent/indexing_scenarios.py` (311 строк)

Day 21: Сквозной сценарий индексации документов и сравнения стратегий chunking.

**Функции:**

- `main() -> None`

### `agent/invariant_scenarios.py` (457 строк)

Day 14: инварианты и ограничения состояния ассистента.

**Классы:**

- `FakeClient` — Имитирует LLM: видит инварианты и умеет отказывать.
  - `complete(self, messages, temperature = None, max_tokens = None) -> Completion`

**Функции:**

- `build_client(config: dict, offline: bool, invariants: Invariants | None) -> object`
- `new_agent(config: dict, offline: bool, invariants: Invariants | None, client: object | None = None) -> Agent` — Создаёт агента с инвариантами в отдельной временной сессии.
- `show_context(agent: Agent) -> None` — Печатает сообщения, которые реально уйдут в модель (инварианты + история).
- `setup_invariants(path: Path) -> Invariants` — Создаёт хранилище инвариантов и наполняет его демо-примерами.
- `run_scenario(config: dict, offline: bool) -> None`
- `run_interactive(config: dict, offline: bool) -> None` — Живой ввод: пользователь сам добавляет инварианты и смотрит отказы.
- `main() -> None`

### `agent/invariants.py` (363 строк)

Инварианты и ограничения состояния агента — то, что ассистент не может нарушать.

**Классы:**

- `GuardVerdict` — Вердикт проверки ответа на нарушение инвариантов.
- `Invariant` — Один инвариант: жёсткое ограничение, которое ассистент не может нарушить.
  - `category_label(self) -> str`
  - `system_line(self) -> str` — Строка для system-сообщения: инвариант в человекочитаемом виде.
  - `summary_line(self) -> str` — Строка для сводки (для вывода пользователю).
  - `to_dict(self) -> dict`
  - `from_dict(cls, data: dict) -> 'Invariant'`
- `Invariants` — Хранилище инвариантов, отделённое от диалога.
  - `path(self) -> Path`
  - `enforce(self) -> bool` — Проверять ли ответ агента на нарушение инвариантов.
  - `enforce(self, value: bool) -> None`
  - `add(self, category: str, description: str, rationale: str = '', invariant_id: str | None = None) -> Invariant` — Добавляет инвариант и возвращает его. id генерируется автоматически.
  - `remove(self, invariant_id: str) -> bool` — Удаляет инвариант по id. Возвращает True, если он был.
  - `get(self, invariant_id: str) -> Invariant | None` — Инвариант по id (None, если не найден).
  - `all(self) -> list[Invariant]` — Все инварианты в порядке добавления.
  - `clear(self) -> None` — Удаляет все инварианты.
  - `system_message(self) -> str` — System-сообщение с инвариантами (подмешивается к каждому запросу).
  - `guard_check_messages(self, user_request: str, assistant_reply: str) -> list[dict]` — Сообщения короткой проверки: нарушает ли ответ инварианты.
  - `refusal_message(self, verdict: GuardVerdict) -> str` — Текст отказа: объясняет, какой инвариант нарушен и почему.
  - `summarize(self) -> str` — Краткая человекочитаемая сводка всех инвариантов.

**Функции:**

- `default_invariants_path(history_dir: Path | None = None) -> Path` — Путь к файлу инвариантов (по умолчанию рядом с историей).
- `parse_guard_verdict(text: str) -> 'GuardVerdict'` — Разбирает ответ модели на проверку нарушения инвариантов.

### `agent/llm_client.py` (95 строк)

HTTP-клиент для запросов к LLM через OpenAI-совместимый API.

**Классы:**

- `Completion` — Ответ LLM: текст и информация об использовании токенов.
- `LLMClient` — Отправляет запросы к API и разбирает ответ.
  - `config(self) -> dict` — Конфигурация клиента, переданная при создании.
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`

### `agent/local_llm.py` (380 строк)

Day 26: локальный запуск LLM через llama.cpp.

**Классы:**

- `LocalLLMError` (RuntimeError) — Ошибка локального запуска LLM (нет бинарника, модели, сервер не поднялся).
- `LocalLLMConfig` — Настройки локальной LLM из секции ``local_llm`` конфигурации.
  - `from_config(cls, config: dict | None) -> LocalLLMConfig`
  - `llama_dir_path(self) -> Path`
  - `model_file(self) -> Path`
  - `binary(self, name: str, explicit: str = '') -> Path | None` — Ищет исполняемый бинарник: явный путь → сборка llama.cpp → PATH.
  - `server_binary(self) -> Path | None`
  - `cli_binary(self) -> Path | None`
- `LocalLLMServer` — Обёртка над процессом ``llama-server`` (OpenAI-совместимый endpoint).
  - `base_url(self) -> str`
  - `is_running(self) -> bool`
  - `build_command(self) -> list[str]`
  - `start(self, wait: bool = True) -> LocalLLMServer` — Запускает сервер и (по умолчанию) ждёт готовности.
  - `wait_ready(self, timeout: float | None = None) -> None` — Опрашивает ``/health`` до готовности или таймаута.
  - `stop(self) -> None` — Останавливает сервер, при необходимости эскалируя до kill.
  - `client_config(self) -> dict` — Конфигурация для :class:`agent.llm_client.LLMClient` (drop-in).
  - `max_context_tokens(self) -> int` — Безопасный лимит контекста клиента (ctx_size минус запас на ответ).
  - `build_client(self) -> LLMClient` — LLM-клиент, направленный на локальный llama-server.
  - `list_models(self) -> list[str]` — Возвращает имена моделей из ``/v1/models``.
  - `complete(self, prompt: str, max_tokens: int = 128) -> Completion` — Одиночный запрос к локальному серверу через OpenAI-совместимый API.
  - `run_cli_prompt(self, prompt: str, max_tokens: int = 64, timeout: float = 180.0) -> str` — Разовый запрос через ``llama-cli`` (проверка «обращения через CLI»).
- `StubLocalLLM` — Детерминированный офлайн-стаб локальной LLM (без бинарников и модели).
  - `base_url(self) -> str`
  - `answer(self, prompt: str) -> str`
  - `complete(self, prompt: str, max_tokens: int = 128) -> Completion`
  - `client_config(self) -> dict`

**Функции:**

- `build_local_client(config: dict) -> tuple[LocalLLMServer, LLMClient]` — Поднимает локальный сервер и возвращает (сервер, клиент) для агентов.

### `agent/local_llm_scenarios.py` (262 строк)

Day 26: сквозной сценарий запуска локальной LLM через llama.cpp.

**Классы:**

- `CheckResult` — Результат одной проверки сценария.

**Функции:**

- `main() -> int`

### `agent/local_rag.py` (1571 строк)

Day 28: RAG-система, полностью работающая локально.

**Классы:**

- `LocalRagStubClient` — Офлайн-клиент локальной LLM, отвечающий строго по RAG-контексту.
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`
- `LocalRagSystem` — Полностью локальный RAG: поиск по FAISS + генерация заданным клиентом.
  - `total_vectors(self) -> int`
  - `retrieve(self, question: str, top_k: int | None = None) -> list[RagSource]` — Локальный поиск релевантных чанков (с фильтром по порогу, без модели).
  - `ask_with_rag(self, question: str, top_k: int | None = None, history: list[dict[str, Any]] | None = None, temperature: float | None = None, max_tokens: int | None = None, context_window: int | None = None, system_prompt: str | None = None) -> RagAnswer` — RAG-запрос с опциональным переопределением параметров генерации.
  - `ask_without_rag(self, question: str) -> RagAnswer`
  - `ask(self, question: str) -> RagAnswer`

**Функции:**

- `is_generic_query(question: str) -> bool` — Определяет общий («что можно приготовить?») запрос по ключевым фразам.
- `has_meta_leak(text: str) -> bool` — Есть ли в ответе упоминание служебного контекста («контекст описывает…»).
- `strip_meta_leak(text: str) -> str` — Удаляет предложения с упоминанием служебного контекста.
- `detect_category(question: str) -> str | None` — Возвращает подстроку категории (в нижнем регистре) для запроса-подборки.
- `is_browse_query(question: str) -> bool` — Запрос-подборка: общий или про категорию («рецепт из говядины»).
- `is_followup_query(question: str) -> bool` — Уточняющий запрос-отсылка к предыдущему ответу («подробней про 2 позицию»).
- `resolve_followup_query(client: Any, question: str, history: list[dict[str, Any]]) -> str | None` — Преобразует уточняющий вопрос-отсылку в название блюда.
- `select_history(messages: list[dict[str, Any]], max_messages: int = 8, max_chars: int = DEFAULT_HISTORY_MAX_CHARS) -> list[dict[str, str]]` — Отбирает последние реплики диалога в пределах окна и бюджета символов.
- `build_local_rag_client(config: dict, offline: bool = False, model_path: str | None = None, port: int | None = None) -> tuple[LocalLLMServer | None, Any]` — Готовит клиент локальной генерации.
- `retrieve_by_category(retriever: RagRetriever, question: str, category: str, limit: int = DEFAULT_CATEGORY_SUGGESTIONS) -> list[RagSource]` — Подбирает разные рецепты категории, ранжируя их по сходству с запросом.
- `retrieve_recipe_sections(retriever: RagRetriever, recipe_id: int | str, question: str = '', limit: int = 32) -> list[RagSource]` — Собирает все секции одного рецепта (описание, ингредиенты, шаги, оборудование).
- `timed_rag_query(client: Any, retriever: RagRetriever, question: str, top_k: int = 5, temperature: float = DEFAULT_TEMPERATURE, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD, history: list[dict[str, Any]] | None = None, generic_threshold: float = DEFAULT_GENERIC_THRESHOLD, generic_min_sources: int = DEFAULT_GENERIC_MIN_SOURCES, category_suggestions: int = DEFAULT_CATEGORY_SUGGESTIONS, history_max_messages: int = 8, history_max_chars: int = DEFAULT_HISTORY_MAX_CHARS, max_tokens: int | None = None, system_prompt: str | None = None) -> dict[str, Any]` — Выполняет RAG-запрос локально и раздельно замеряет этапы.
- `answer_self_consistency(texts: list[str]) -> dict[str, Any]` — Оценивает согласованность нескольких ответов модели на один вопрос.
- `compare_local_vs_cloud(retriever: RagRetriever, local_client: Any, cloud_client: Any | None = None, questions: list[dict[str, Any]] | None = None, top_k: int = 5, temperature: float = DEFAULT_TEMPERATURE, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS, repeats: int = 1, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD, local_label: str = 'local-llm', cloud_label: str = 'cloud', progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]` — Сравнивает локальную и облачную модель на контрольных вопросах.
- `render_local_rag_markdown(report: dict[str, Any]) -> str` — Формирует Markdown-отчёт сравнения локальной и облачной модели.

### `agent/local_rag_scenarios.py` (696 строк)

Day 28: сквозной сценарий RAG-системы, полностью работающей локально.

**Классы:**

- `CheckResult` — Результат одной HTTP-проверки веб-чата с RAG.

**Функции:**

- `main() -> int`

### `agent/mcp_client.py` (237 строк)

Day 16: подключение к MCP и получение списка инструментов.

**Классы:**

- `McpTool` — Инструмент, объявленный MCP-сервером.
- `McpConnection` — Результат подключения: сервер и его инструменты.

**Функции:**

- `async connect_and_list_tools(url: str = DEFAULT_MCP_URL, user_agent: str = BROWSER_USER_AGENT, timeout: float = 30.0) -> McpConnection` — Устанавливает MCP-соединение и возвращает список инструментов.
- `print_connection(url: str, connection: McpConnection) -> None` — Печатает состояние соединения и список инструментов.
- `verify_connection(connection: McpConnection) -> list[tuple[str, bool]]` — Проверяет, что соединение установлено, а инструменты вернулись корректно.
- `connection_as_dict(connection: McpConnection) -> dict` — Представление результата для машинного вывода (JSON).
- `load_mcp_config() -> dict` — Читает секцию ``mcp`` конфигурации, если она доступна.
- `main() -> None`

### `agent/mcp_tools.py` (297 строк)

Day 17: MCP-инструменты в агенте.

**Классы:**

- `ToolSpec` — Инструмент, объявленный MCP-сервером.
- `ToolCall` — Выбранный агентом вызов инструмента: имя и аргументы.
- `ToolResult` — Результат вызова MCP-инструмента.
- `McpToolRuntime` — Клиент MCP-сервера: список инструментов и их вызов.
  - `command(self) -> str`
  - `args(self) -> list[str]`
  - `list_tools(self, refresh: bool = False) -> list[ToolSpec]` — Возвращает инструменты MCP-сервера (с кэшированием).
  - `call(self, name: str, arguments: dict | None = None) -> ToolResult` — Вызывает MCP-инструмент по имени и возвращает результат.
  - `has_tool(self, name: str) -> bool`
  - `summarize(self) -> str` — Человекочитаемое описание инструментов (для CLI/сценария).
  - `plan_messages(self, user_request: str, results: list[dict] | None = None) -> list[dict]` — Сообщения для выбора инструмента и аргументов под запрос.
  - `parse_plan(text: str) -> ToolCall | None` — Разбирает JSON-план модели. None — инструмент вызывать не нужно.
  - `result_message(self, result: ToolResult) -> dict` — System-сообщение с результатом вызова, которое видит основная модель.

**Функции:**

- `runtime_from_config(config: dict) -> McpToolRuntime | None` — Создаёт MCP-рантайм из секции ``tools`` конфигурации (None — выключено).

### `agent/memory.py` (279 строк)

Модель памяти агента: три отдельных слоя с явным выбором, что куда сохранять.

**Классы:**

- `LongTermKind` (str, Enum) — Категория записи долговременной памяти.
- `WorkingMemory` — Слой рабочей памяти: данные текущей задачи (в рамках сессии).
  - `all(self) -> dict[str, str]` — Все записи рабочей памяти текущей сессии.
  - `get(self, key: str) -> str | None` — Одна запись рабочей памяти.
  - `remember(self, key: str, value: str) -> None` — Явно сохраняет запись в рабочую память.
  - `remember_many(self, items: dict[str, str]) -> None` — Сохраняет несколько записей в рабочую память.
  - `clear(self) -> None` — Очищает рабочую память (например, при смене текущей задачи).
- `LongTermMemory` — Слой долговременной памяти: профиль, решения, знания.
  - `path(self) -> Path` — Файл, в котором хранится долговременная память.
  - `remember(self, kind: str, key: str, value: str) -> None` — Явно сохраняет запись в долговременную память нужной категории.
  - `get(self, kind: str, key: str) -> str | None` — Одна запись долговременной памяти.
  - `all(self, kind: str | None = None) -> dict[str, dict[str, str]]` — Записи долговременной памяти (все категории или одна).
  - `forget(self, kind: str, key: str) -> bool` — Удаляет запись из долговременной памяти.
  - `clear(self) -> None` — Полностью очищает долговременную память.
- `MemoryLayers` — Модель памяти агента: три слоя + сборка общего контекста для модели.
  - `short_term(self) -> list[dict]` — Краткосрочная память: сообщения текущего диалога.
  - `remember_short_term(self, role: str, content: str) -> None` — Добавляет сообщение в краткосрочную память (текущий диалог).
  - `working(self) -> WorkingMemory` — Рабочая память: данные текущей задачи.
  - `remember_working(self, key: str, value: str) -> None` — Явно сохраняет запись в рабочую память (текущая задача).
  - `long_term(self) -> LongTermMemory` — Долговременная память: профиль, решения, знания.
  - `remember_long_term(self, kind: str, key: str, value: str) -> None` — Явно сохраняет запись в долговременную память нужной категории.
  - `build_memory_context(self, include: Iterable[str] = LONG_TERM_KINDS) -> list[dict]` — Собирает блоки памяти в список system-сообщений для модели.
  - `summarize(self) -> str` — Краткая сводка всех слоёв памяти (для вывода пользователю/логов).

**Функции:**

- `default_profile_path(history_dir: Path | None = None) -> Path` — Путь к файлу долговременной памяти (по умолчанию рядом с историей).

### `agent/memory_scenarios.py` (299 строк)

Day 11: модель памяти агента — три отдельных слоя.

**Классы:**

- `FakeClient` — Имитирует LLM: считает токены локально и отражает память в ответе.
  - `complete(self, messages, temperature = None, max_tokens = None) -> Completion`

**Функции:**

- `build_client(config: dict, offline: bool) -> object`
- `profile_path_from_config(config: dict) -> Path` — Путь к файлу долговременной памяти из конфигурации (или по умолчанию).
- `new_agent(config: dict, offline: bool) -> tuple[Agent, Path]` — Создаёт агента с моделью памяти; долговременная память живёт в постоянном файле.
- `show_memory(agent: Agent) -> None` — Печатает содержимое всех слоёв памяти.
- `show_context(agent: Agent) -> None` — Печатает сообщения, которые реально уйдут в модель (с блоками памяти).
- `demo(config: dict, offline: bool, turns: list[str], long_term_path: Path | None = None) -> None` — Прогоняет сценарий и в конце показывает, что в каком слое осталось.
- `run_interactive(config: dict, offline: bool) -> None` — Живой ввод: пользователь сам решает, что и в какой слой сохранять.
- `main() -> None`

### `agent/optimization.py` (995 строк)

Day 29: оптимизация локальной LLM под конкретную задачу.

**Классы:**

- `GenerationParams` — Параметры генерации одной конфигурации (оптимизируемые в вебе).
  - `from_dict(cls, data: dict[str, Any] | None) -> GenerationParams`
  - `to_dict(self) -> dict[str, Any]`
- `ModelVariant` — Один вариант локальной модели (квантование GGUF).
  - `from_dict(cls, data: dict[str, Any]) -> ModelVariant`
  - `path(self) -> Path`
  - `available(self) -> bool` — Есть ли файл модели на диске (пустой путь — недоступен).
  - `actual_size_mb(self) -> float | None` — Реальный размер файла, если он есть; иначе объявленный ``size_mb``.
  - `to_dict(self) -> dict[str, Any]`
- `PromptTemplate` — System-промпт конфигурации (базовый или оптимизированный под кейс).
  - `to_dict(self) -> dict[str, str]`
- `OptimizationStubClient` — Офлайн-клиент локальной LLM, отвечающий строго по RAG-контексту.
  - `tokens_per_sec(self) -> float`
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`
- `ConfigurationSpec` — Описание одной конфигурации оптимизации (из секции ``optimization``).
  - `from_dict(cls, data: dict[str, Any]) -> ConfigurationSpec`
- `OptimizationConfiguration` — Готовая к прогону конфигурация: параметры + шаблон + клиент/вариант.
  - `resources(self) -> dict[str, Any]`
  - `to_dict(self) -> dict[str, Any]`

**Функции:**

- `parse_presets(config: dict | None) -> dict[str, GenerationParams]` — Читает пресеты из ``local_llm.presets`` (дополняя значения по умолчанию).
- `parse_variants(config: dict | None) -> list[ModelVariant]` — Читает варианты моделей из ``local_llm.variants`` (иначе — по умолчанию).
- `find_variant(variants: list[ModelVariant], label: str | None) -> ModelVariant | None` — Находит вариант по метке (без учёта регистра); иначе первый доступный/первый.
- `find_template(name: str | None) -> PromptTemplate` — Возвращает шаблон по имени (по умолчанию — базовый).
- `measure_peak_rss_mb() -> float | None` — Пиковое потребление RAM дочерними процессами (llama-server), МБ.
- `parse_configuration_specs(config: dict | None) -> list[ConfigurationSpec]` — Читает список конфигураций из ``optimization.configurations``.
- `compare_optimizations(retriever: RagRetriever, configurations: list[OptimizationConfiguration], questions: list[dict[str, Any]] | None = None, top_k: int = 5, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD, repeats: int = 1, baseline_name: str = 'baseline', optimized_name: str = 'optimized', progress_callback: Callable[[str, int, int, str], None] | None = None) -> dict[str, Any]` — Прогоняет несколько конфигураций и считает дельту «до/после».
- `render_optimization_markdown(report: dict[str, Any]) -> str` — Формирует Markdown-отчёт сравнения «до/после» оптимизации.

### `agent/optimization_scenarios.py` (470 строк)

Day 29: сквозной сценарий оптимизации локальной LLM под задачу (рецепты).

**Классы:**

- `CheckResult` — Результат одной HTTP-проверки веб-интерфейса.

**Функции:**

- `main() -> int`

### `agent/orchestration_scenarios.py` (494 строк)

Day 20: сквозной сценарий оркестрации нескольких MCP-серверов.

**Классы:**

- `ScriptedLLMClient` — Заглушка LLM для ``--offline``: заранее заданный план вызовов.
  - `complete(self, messages, temperature = None, max_tokens = None) -> Completion`

**Функции:**

- `main() -> None`

### `agent/orchestrator.py` (613 строк)

Day 20: оркестрация нескольких MCP-серверов.

**Классы:**

- `ToolRoutingError` (Exception) — Вызов не удалось сопоставить ни одному инструменту известных серверов.
- `ServerSpec` — Описание одного MCP-сервера из конфигурации.
  - `build_runtime(self) -> McpToolRuntime` — Создаёт клиент MCP-сервера (stdio) по этому описанию.
- `RoutedTool` — Инструмент с указанием сервера, который его объявляет.
  - `name(self) -> str` — Короткое имя инструмента без сервера.
  - `qualified(self) -> str` — Полное имя вида ``сервер.инструмент``.
  - `description(self) -> str`
  - `input_schema(self) -> dict`
- `StaticToolRuntime` — Рантайм-заглушка без MCP: фиксированные инструменты и их обработчики.
  - `list_tools(self, refresh: bool = False) -> list[ToolSpec]`
  - `has_tool(self, name: str) -> bool`
  - `call(self, name: str, arguments: dict | None = None) -> ToolResult`
- `McpOrchestrator` — Регистрирует несколько MCP-серверов и маршрутизирует вызовы инструментов.
  - `server_names(self) -> list[str]`
  - `runtime(self, server: str)`
  - `list_tools(self, server: str, refresh: bool = False) -> list[ToolSpec]`
  - `list_routes(self, refresh: bool = False) -> list[RoutedTool]` — Все инструменты всех серверов с меткой сервера.
  - `catalog(self, refresh: bool = False) -> list[dict]` — Каталог инструментов для LLM: сервер, имя, описание и параметры.
  - `resolve(self, tool: str) -> RoutedTool` — Находит, какой сервер объявляет инструмент ``tool``.
  - `has_tool(self, tool: str) -> bool`
  - `call(self, tool: str, arguments: dict | None = None) -> ToolResult` — Маршрутизирует вызов инструмента на нужный сервер и выполняет его.
  - `summarize(self) -> str` — Человекочитаемый список серверов и их инструментов.
  - `plan_messages(self, user_request: str, results: list[dict] | None = None) -> list[dict]` — Сообщения для выбора СЛЕДУЮЩЕГО инструмента под запрос.
  - `parse_plan(text: str) -> ToolCall | None` — Разбирает JSON-план модели. None — инструменты больше не нужны.
  - `result_message(result: ToolResult) -> dict` — System-сообщение с результатом вызова (видно основной модели).

**Функции:**

- `orchestrator_from_config(config: dict) -> McpOrchestrator | None` — Создаёт оркестратор из секции ``orchestrator`` (None — выключено).
- `verify_agent_flow(trace: list[ToolCall], results: list[ToolResult], required_servers: list[str] | None = None, min_servers: int = 2) -> list[str]` — Проверяет длинный флоу агента по разным MCP-серверам.
- `describe_flow(trace: list[ToolCall], results: list[ToolResult]) -> list[str]` — Пошаговое описание флоу: номер, сервер, инструмент и ключи результата.

### `agent/personalization.py` (283 строк)

Персонализация ассистента поверх модели памяти.

**Классы:**

- `UserProfile` — Профиль пользователя: личность + предпочтения (стиль/формат/ограничения).
  - `from_dict(cls, data: dict) -> 'UserProfile'` — Восстанавливает профиль из словаря (например, из JSON-файла).
  - `to_dict(self) -> dict` — Сериализует профиль в словарь для сохранения в JSON.
  - `set_preference(self, group: str, key: str, value: str) -> None` — Записывает одно предпочтение в нужную группу (style/format/constraints).
  - `remove_preference(self, group: str, key: str) -> bool` — Удаляет предпочтение. Возвращает True, если оно было.
  - `system_message(self) -> str` — Собирает system-сообщение профиля для подстановки в запрос.
  - `is_empty(self) -> bool` — Пустой ли профиль (нечего подмешивать в запрос).
- `Personalization` — Управляет профилями пользователя и активным профилем.
  - `path(self) -> Path` — Файл, в котором хранятся профили.
  - `active_name(self) -> str | None` — Имя активного профиля (None — персонализация не активна).
  - `profiles(self) -> dict[str, UserProfile]` — Все сохранённые профили: имя -> профиль.
  - `has(self, name: str) -> bool` — Есть ли профиль с таким именем.
  - `get(self, name: str) -> UserProfile | None` — Профиль по имени (None, если такого нет).
  - `save(self, name: str, profile: UserProfile) -> None` — Сохраняет профиль под указанным именем.
  - `delete(self, name: str) -> bool` — Удаляет профиль. Возвращает True, если он был.
  - `set_active(self, name: str | None) -> bool` — Делает профиль активным (None — отключить персонализацию).
  - `active(self) -> UserProfile | None` — Активный профиль (None — персонализация не активна).
  - `system_message(self) -> str` — System-сообщение активного профиля (пусто, если профиля нет/пуст).
  - `summarize(self) -> str` — Краткое описание всех профилей и активного (для вывода пользователю).

**Функции:**

- `default_profiles_path(history_dir: Path | None = None) -> Path` — Путь к файлу профилей (по умолчанию рядом с историей).

### `agent/personalization_scenarios.py` (324 строк)

Day 12: персонализация ассистента поверх модели памяти.

**Классы:**

- `FakeClient` — Имитирует LLM: показывает, какой профиль и сколько сообщений ушло в модель.
  - `complete(self, messages, temperature = None, max_tokens = None) -> Completion`

**Функции:**

- `build_client(config: dict, offline: bool) -> object`
- `profiles_path_from_config(config: dict) -> Path` — Путь к файлу профилей из конфигурации (или по умолчанию).
- `new_agent(config: dict, offline: bool, personalization: Personalization) -> Agent` — Создаёт агента с персонализацией в отдельной временной сессии.
- `show_context(agent: Agent) -> None` — Печатает сообщения, которые реально уйдут в модель (профиль + история).
- `run_scenario(config: dict, offline: bool, query: str) -> None` — Прогоняет один вопрос через каждый профиль и показывает результат.
- `run_interactive(config: dict, offline: bool) -> None` — Живой ввод: пользователь сам переключает профили и смотрит контекст.
- `main() -> None`

### `agent/pipeline.py` (392 строк)

Day 19: движок композиции MCP-инструментов.

**Классы:**

- `PipelineStep` — Один шаг пайплайна: какой инструмент и с какими аргументами вызвать.
- `StepResult` — Результат выполнения шага: фактические аргументы и ответ инструмента.
  - `is_error(self) -> bool`
  - `output(self) -> dict` — Структурированный результат шага (пустой, если его нет).
- `PipelineRun` — Результат всего пайплайна: шаги по порядку и итоговый выход.
  - `ok(self) -> bool` — Все шаги выполнены без ошибок и пайплайн не пуст.
  - `final(self) -> dict` — Структурированный выход последнего шага.
  - `saved_path(self) -> str | None` — Путь сохранённого файла (из последнего шага, если он пишет файл).
  - `step(self, name: str) -> StepResult | None` — Шаг по имени (None, если такого шага нет).
  - `summary(self) -> str` — Человекочитаемая сводка по шагам пайплайна.
- `Pipeline` — Пайплайн из MCP-инструментов, выполняемый автоматически по шагам.
  - `run(self, runtime: McpToolRuntime, initial: dict | None = None) -> PipelineRun` — Выполняет шаги по порядку, передавая выходы в следующие аргументы.

**Функции:**

- `resolve_refs(value, context: dict)` — Подставляет ``${step.path}`` в аргументы, используя результаты шагов.
- `gitflic_summary_pipeline(query: str, output_path: str = 'history/pipeline_result.md', size: int = 5, max_sentences: int = 3) -> Pipeline` — Собирает пайплайн ``search → summarize → saveToFile`` для GitFlic.
- `verify_pipeline_run(run: PipelineRun, check_file: bool = True) -> list[str]` — Проверяет автоматическое выполнение и передачу данных между шагами.
- `pipeline_runtime_from_config(config: dict) -> McpToolRuntime | None` — Создаёт рантайм пайплайна из секции ``pipeline`` конфигурации.

### `agent/pipeline_scenarios.py` (255 строк)

Day 19: сквозной сценарий композиции MCP-инструментов.

**Функции:**

- `main() -> None`

### `agent/project_docs.py` (404 строк)

День 31: индексация документации проекта для ассистента разработчика.

**Классы:**

- `ProjectDocument` — Один документ проекта, попадающий в RAG-индекс.

**Функции:**

- `discover_documents(root: Path = PROJECT_ROOT, globs: tuple[str, ...] = DEFAULT_DOC_GLOBS) -> list[Path]` — Находит существующие markdown-файлы документации (отсортировано).
- `load_document(path: Path, root: Path = PROJECT_ROOT) -> ProjectDocument` — Загружает один markdown-документ в :class:`ProjectDocument`.
- `load_documents(root: Path = PROJECT_ROOT, globs: tuple[str, ...] = DEFAULT_DOC_GLOBS) -> list[ProjectDocument]` — Загружает всю документацию проекта.
- `chunk_markdown(doc: ProjectDocument, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Chunk]` — Структурный чанкинг markdown по заголовкам.
- `build_documents_index(root: Path = PROJECT_ROOT, index_dir: str | Path = DEFAULT_INDEX_DIR, dataset: str = DEFAULT_DATASET, strategy: str = DEFAULT_STRATEGY, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP, embedder: HashingEmbedder | None = None, documents: list[ProjectDocument] | None = None, save: bool = True) -> tuple[FaissIndex, list[ProjectDocument]]` — Собирает (и при ``save=True`` сохраняет) индекс документации проекта.
- `load_project_retriever(index_dir: str | Path = DEFAULT_INDEX_DIR, dataset: str = DEFAULT_DATASET, strategy: str = DEFAULT_STRATEGY, top_k: int = 5, min_score: float = 0.0, root: Path = PROJECT_ROOT) -> RagRetriever` — Загружает индекс документации с диска как :class:`RagRetriever`.
- `build_docs_context(sources: list[RagSource], max_chars: int = 6000) -> str` — Собирает контекст документации с указанием файла и раздела.
- `main() -> int`

### `agent/rag.py` (939 строк)

Day 22: Модуль RAG (Retrieval-Augmented Generation) для рецептов гриля и BBQ.

**Классы:**

- `RagSource` — Найденный чанк-источник для формирования контекста RAG.
  - `to_dict(self) -> dict[str, Any]`
  - `from_search_result(cls, result: dict[str, Any]) -> RagSource`
- `RagAnswer` — Результат выполнения запроса (с RAG или без RAG).
  - `to_dict(self) -> dict[str, Any]`
- `RagRetriever` — Извлекает релевантные чанки из FAISS-индекса и собирает контекст.
  - `total_vectors(self) -> int`
  - `retrieve(self, query: str, top_k: int | None = None) -> list[RagSource]` — Ищет ближайшие чанки по запросу и оборачивает в RagSource.
  - `build_context(self, sources: list[RagSource], max_chars: int = 6000) -> str` — Собирает структурированный текст контекста с нумерацией источников.
- `RagAgent` — Агент с переключаемыми режимами: с RAG (retrieval) и без RAG (no_rag).
  - `mode(self) -> str` — Текущий активный режим агента ('rag' или 'no_rag').
  - `set_mode(self, mode: str) -> None` — Переключает режим работы агента.
  - `ask(self, question: str) -> RagAnswer` — Отправляет вопрос модели в соответствии с текущим режимом.
  - `ask_with_rag(self, question: str, top_k: int | None = None) -> RagAnswer` — Выполняет запрос с RAG независимо от текущего режима.
  - `ask_without_rag(self, question: str) -> RagAnswer` — Выполняет прямой запрос к LLM (без RAG) независимо от режима.

**Функции:**

- `build_rag_messages(question: str, sources: list[RagSource], max_chars: int = 6000, history: list[dict[str, str]] | None = None, extra_instruction: str | None = None, system_prompt: str | None = None) -> list[dict[str, str]]` — Формирует список сообщений (messages) для RAG-запроса.
- `build_plain_messages(question: str) -> list[dict[str, str]]` — Формирует список сообщений для baseline-запроса (без RAG).
- `rag_query(client: LLMClient, retriever: RagRetriever, question: str, top_k: int = 5, temperature: float = 0.2, max_context_chars: int = 6000) -> RagAnswer` — Полный пайплайн RAG-запроса:
- `plain_query(client: LLMClient, question: str, temperature: float = 0.2) -> RagAnswer` — Запрос к LLM напрямую без использования RAG (Baseline).
- `load_retriever(index_dir: str | Path, dataset: str = 'recipt_all', strategy: str = 'structural', embedder: Embedder | None = None, top_k: int = 5) -> RagRetriever` — Загружает FAISS-индекс с диска и создает RagRetriever.
- `evaluate_answer_facts(answer_text: str, expected_facts: list[list[str]]) -> dict[str, Any]` — Проверяет покрытие ключевых фактов/синонимов в ответе модели.
- `evaluate_source_recall(sources: list[RagSource], expected_ids: list[int | str]) -> dict[str, Any]` — Проверяет, попал ли целевой рецепт в найденные источники.
- `evaluate_citations(answer_text: str, sources: list[RagSource]) -> dict[str, Any]` — Проверяет наличие ссылок на источники в ответе модели.
- `evaluate_rag_comparison(agent: RagAgent, questions: list[dict[str, Any]] | None = None, progress_callback: Callable[[int, int, str], None] | None = None, run_llm: bool = True) -> dict[str, Any]` — Выполняет пакетное сравнение двух режимов по набору контрольных вопросов.
- `render_rag_comparison_markdown(report: dict[str, Any]) -> str` — Формирует наглядный Markdown-отчёт со сравнением RAG и Baseline.

### `agent/rag_chat.py` (812 строк)

RAG-чат с памятью задачи (Task State) и выводом источников с URL.

**Классы:**

- `ChatSource` — Источник, использованный для ответа (chunk_id + название + раздел + URL).
  - `to_dict(self) -> dict[str, Any]`
  - `from_rag_source(cls, src: RagSource) -> ChatSource`
  - `format_line(self, index: int) -> str` — Форматирует строку источника для вывода пользователю.
- `RagTaskState` — Память задачи (task state) для удержания контекста длинных диалогов.
  - `to_dict(self) -> dict[str, Any]`
  - `from_dict(cls, data: dict[str, Any] | None) -> RagTaskState`
  - `set_goal(self, goal: str) -> None`
  - `add_clarification(self, key: str, value: str) -> None`
  - `add_constraint(self, constraint: str) -> None`
  - `add_term(self, term: str, definition: str) -> None`
  - `render_summary(self) -> str` — Формирует текстовое резюме состояния задачи для промпта LLM.
  - `render_banner(self) -> str` — Компактная карточка состояния задачи для CLI.
  - `extract_from_user_turn(self, user_msg: str) -> dict[str, Any]` — Эвристически распознает цель, ограничения, термины и параметры из реплики.
  - `check_constraint_violation(self, user_msg: str) -> str | None` — Проверяет, не нарушает ли запрос пользователя активные ограничения.
- `RagChatSession` — Сессия чата: оборачивает Conversation и сохраняет RagTaskState на диск.
  - `session_id(self) -> str`
  - `messages(self) -> list[dict[str, str]]`
  - `save(self) -> None` — Сохраняет состояние задачи в сессию.
  - `append_turn(self, user_msg: str, assistant_msg: str, sources: list[ChatSource]) -> None` — Добавляет пару реплик диалога в сессию.
  - `clear(self) -> None` — Сбрасывает историю и состояние сессии.
- `RagChatResponse` — Результат ответа RAG-агента в чате.
  - `full_reply_text(self) -> str` — Полный текст ответа с прикреплённым блоком источников.
  - `to_dict(self) -> dict[str, Any]`
- `RagChatAgent` — Агент чата с RAG-поиском, заземлением и памятью задачи.
  - `ask(self, user_message: str, session: RagChatSession, override_query: str | None = None) -> RagChatResponse` — Обрабатывает ход пользователя, обновляет состояние, ищет в RAG и формирует ответ.

**Функции:**

- `build_rag_chat_agent(config: dict[str, Any], top_k: int | None = None, dataset: str | None = None, strategy: str | None = None, index_dir: Path | str | None = None, no_llm: bool = False) -> RagChatAgent` — Создаёт и конфигурирует RagChatAgent из конфигурации проекта.

### `agent/rag_chat_cli.py` (215 строк)

Интерактивный терминальный CLI-чат с RAG и памятью задачи (Task State).

**Функции:**

- `print_banner(session: RagChatSession) -> None` — Выводит стартовую заставку и состояние сессии.
- `handle_command(cmd: str, session: RagChatSession) -> bool` — Обрабатывает slash-команды CLI.
- `main() -> None`

### `agent/rag_chat_scenarios.py` (515 строк)

Day 24 / RAG Chat: Сквозные проверочные сценарии диалога с RAG и памятью задачи.

**Функции:**

- `run_scenario(scenario_id: str, scenario_title: str, steps: list[dict[str, Any]], agent: RagChatAgent, history_dir: Path, max_steps: int | None = None) -> dict[str, Any]` — Выполняет один многошаговый сценарий и возвращает метрики.
- `render_markdown_report(report_data: dict[str, Any]) -> str` — Генерирует подробный Markdown-отчёт по результатам проверки 2 сценариев.
- `main() -> None`

### `agent/rag_scenarios.py` (345 строк)

Day 22: Сквозной сценарий первого RAG-запроса и сравнения с Baseline.

**Функции:**

- `main() -> None`

### `agent/reranking.py` (1110 строк)

Day 23: Модуль реранкинга, фильтрации релевантности и переписывания запросов (Query Rewrite).

**Классы:**

- `RankedSource` — Чанк с метаданными первого (retrieval) и второго (reranking/filter) этапов.
  - `to_dict(self) -> dict[str, Any]`
- `RerankOutcome` — Полный результат выполнения двухэтапного пайплайна.
  - `to_dict(self) -> dict[str, Any]`
- `HeuristicQueryRewriter` — Офлайн-переписыватель запроса: очистка от шума и фокус на ключевых терминах.
  - `rewrite(self, query: str) -> str`
- `LLMQueryRewriter` — Переписыватель запросов через LLM: формирует целевой поисковый запрос.
  - `rewrite(self, query: str) -> str`
- `RelevanceFilter` — Фильтрует кандидатов по порогу сходства и относительной границе.
  - `apply(self, candidates: list[RankedSource]) -> tuple[list[RankedSource], list[RankedSource]]` — Разделяет кандидатов на оставленных (passed) и отсеянных (dropped).
- `HeuristicReranker` — Эвристический реранкер: комбинирует векторное сходство, лексическое покрытие
  - `compute_lexical_score(self, query_tokens: set[str], text: str) -> float` — Считает долю токенов запроса, присутствующих в тексте чанка.
  - `compute_title_score(self, query_tokens: set[str], title: str) -> float` — Считает совпадение токенов запроса с заголовком рецепта.
  - `compute_section_score(self, query: str, section: str) -> float` — Поощряет соответствие раздела интент-маркерам запроса.
  - `rerank(self, candidates: list[RankedSource], query: str) -> list[RankedSource]` — Вычисляет rerank_score для каждого кандидата и сортирует по убыванию.
- `LLMReranker` — Реранкер на основе отдельной LLM-модели (cross-encoder подход через промпт).
  - `rerank(self, candidates: list[RankedSource], query: str) -> list[RankedSource]`
- `RerankPipeline` — Полный двухэтапный пайплайн:
  - `run(self, query: str) -> RerankOutcome` — Выполняет полный пайплайн от запроса до финального списка чанков.
- `RerankRagAnswer` (RagAnswer) — Ответ RAG с подробной информацией о реранкинге и фильтрации.
  - `to_dict(self) -> dict[str, Any]`

**Функции:**

- `rewrite_query(query: str, method: str = 'heuristic', client: LLMClient | None = None) -> str` — Единая функция переписывания запроса по выбранному методу.
- `rag_query_with_pipeline(client: LLMClient, pipeline: RerankPipeline, question: str, pipeline_mode: str = 'full', temperature: float = 0.2, max_context_chars: int = 6000) -> RerankRagAnswer` — Выполняет RAG-запрос через двухэтапный пайплайн с реранкингом и фильтром.
- `evaluate_precision_at_k(sources: list[RagSource], expected_ids: list[int | str]) -> float` — Точность в топ-K (Precision@K): доля чанков, принадлежащих целевому рецепту.
- `evaluate_noise_filter(outcome: RerankOutcome, expected_ids: list[int | str]) -> dict[str, Any]` — Оценивает корректность фильтрации шума.
- `build_pipeline_modes(retriever: RagRetriever, client: LLMClient | None = None, retrieve_k: int = 15, final_k: int = 5, min_score: float = 0.33, rerank_model: str | None = None) -> dict[str, RerankPipeline]` — Фабрика стандартных конфигураций для сравнительного тестирования режимов.
- `evaluate_rerank_comparison(retriever: RagRetriever, client: LLMClient | None = None, modes: list[str] | None = None, questions: list[dict[str, Any]] | None = None, retrieve_k: int = 15, final_k: int = 5, min_score: float = 0.33, run_llm: bool = True, progress_callback: Callable[[int, int, str, str], None] | None = None) -> dict[str, Any]` — Сравнивает качество работы RAG в разных режимах фильтрации/реранкинга.
- `render_rerank_comparison_markdown(report: dict[str, Any]) -> str` — Формирует структурированный Markdown-отчёт по результатам сравнения режимов.

### `agent/reranking_scenarios.py` (414 строк)

Day 23: Сквозной сценарий реранкинга, фильтрации релевантности и Query Rewrite.

**Функции:**

- `main() -> None`

### `agent/review_cli.py` (296 строк)

День 32: CLI автоматического ревью pull request.

**Функции:**

- `now_iso() -> str`
- `load_event(path: str | None) -> dict` — Загружает payload GitHub-события (если путь задан и файл существует).
- `event_pr_info(event: dict) -> dict` — Извлекает из события PR нужные для запуска и публикации поля.
- `post_pr_comment(body: str, repo: str, number: int, token: str) -> tuple[bool, str]` — Публикует комментарий к PR через GitHub REST API (без внешних зависимостей).
- `should_fail(review: CodeReview, fail_on: str) -> bool` — Решает, должен ли пайплайн упасть по итогам ревью.
- `build_review_header(review: CodeReview, mode: str) -> str`
- `main() -> int`

### `agent/review_scenarios.py` (477 строк)

День 32: сквозной сценарий автоматического ревью кода (офлайн-проверка).

**Функции:**

- `now_iso() -> str`
- `check_indexes(index_dir: str, strategy: str) -> dict` — Собирает индексы документации и кода, проверяет состав.
- `check_mcp(runtime, base: str | None, head: str | None) -> dict` — Проверяет MCP-сервер ревью: diff и изменённые файлы по реальному репозиторию.
- `check_heuristics() -> dict` — Прямая проверка эвристик и парсера находок.
- `build_fixture_repo() -> tuple[str, str]` — Создаёт временный git-репозиторий с «посаженным» патчем.
- `check_review_e2e(config: dict, index_dir: str, strategy: str, top_k: int) -> dict` — Полный прогон ассистента на временном репозитории с патчем.
- `render_report(index_info: dict, mcp_info: dict, heuristics: dict, e2e: dict, all_ok: bool) -> str`
- `main() -> int`

### `agent/scheduler_scenarios.py` (213 строк)

Day 18: планировщик и фоновые задачи — подсчёт публичных проектов GitFlic.

**Функции:**

- `main() -> None`

### `agent/support_assistant.py` (789 строк)

День 33: ассистент поддержки пользователей.

**Классы:**

- `SupportAnswer` — Ответ ассистента поддержки.
  - `to_dict(self) -> dict[str, Any]`
- `SupportStubClient` — Детерминированный офлайн-клиент ассистента поддержки.
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`
- `SupportAssistant` — Ассистент поддержки: RAG по FAQ + контекст тикета/пользователя через MCP.
  - `total_vectors(self) -> int`
  - `retrieve(self, question: str, top_k: int | None = None) -> list[RagSource]` — Ищет релевантные карточки FAQ.
  - `resolve_ticket(self, question: str, ticket_id: str | None = None, user_id: int | None = None) -> tuple[str | None, int | None]` — Определяет тикет/пользователя для вопроса (через MCP).
  - `collect_ticket_context(self, ticket_id: str | None, user_id: int | None = None) -> dict[str, Any]` — Собирает контекст тикета и пользователя через MCP.
  - `ask(self, question: str, ticket_id: str | None = None, user_id: int | None = None, top_k: int | None = None, history: list[dict[str, str]] | None = None) -> SupportAnswer` — Отвечает на вопрос о продукте с учётом FAQ и данных тикета.
  - `get_ticket(self, ticket_id: str) -> dict[str, Any]`
  - `get_user(self, user_id: int) -> dict[str, Any]`
  - `list_tickets(self, user_id: int | None = None) -> dict[str, Any]`

**Функции:**

- `render_ticket_context(context: dict[str, Any]) -> str` — Человекочитаемый контекст тикета и пользователя для промпта.
- `build_support_messages(question: str, faq_sources: list[RagSource], ticket_context: dict[str, Any], max_chars: int = 6000, history: list[dict[str, str]] | None = None, system_prompt: str | None = None) -> list[dict[str, str]]` — Формирует сообщения ассистента: FAQ + контекст тикета + вопрос.
- `parse_answer_meta(text: str) -> dict[str, Any]` — Разбирает машиночитаемый JSON-блок ответа (answer_kind/ticket_id/escalate).
- `resolve_support_ticket(question: str, ticket_id: str | None, user_id: int | None, call_tool: Any) -> tuple[str | None, int | None]` — Определяет тикет по вопросу: явный id → тикет пользователя → текстовый поиск.
- `build_support_runtime(config: dict | None = None) -> McpToolRuntime | None` — Создаёт MCP-рантайм сервера поддержки из конфигурации.
- `build_support_assistant(config: dict | None = None, offline: bool = False, faq_path: str = DEFAULT_FAQ_PATH, index_dir: str = DEFAULT_INDEX_DIR, dataset: str = DEFAULT_DATASET, strategy: str = DEFAULT_STRATEGY, top_k: int = 5, min_score: float = 0.0, temperature: float = 0.1, max_context_chars: int = 6000, relevance_threshold: float = 0.3, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP, build_index_if_missing: bool = True, with_support_mcp: bool = True) -> SupportAssistant` — Собирает ассистента поддержки: FAQ-индекс, MCP-рантайм, LLM-клиент.

### `agent/support_cli.py` (348 строк)

День 33: интерактивный CLI ассистента поддержки.

**Функции:**

- `print_banner(assistant) -> None`
- `handle_command(raw: str, assistant, state: dict) -> bool` — Обрабатывает slash-команду. True — ввод был командой.
- `main() -> int`

### `agent/support_faq.py` (493 строк)

День 33: RAG по FAQ ассистента поддержки.

**Классы:**

- `SupportFaq` — Одна карточка FAQ, попадающая в RAG-индекс.

**Функции:**

- `load_faq(path: str | Path = DEFAULT_FAQ_PATH, root: Path = PROJECT_ROOT) -> list[SupportFaq]` — Загружает карточки FAQ из JSON-файла.
- `chunk_faq(item: SupportFaq, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Chunk]` — Превращает карточку FAQ в один или несколько чанков.
- `build_faq_index(root: Path = PROJECT_ROOT, faq_path: str | Path = DEFAULT_FAQ_PATH, index_dir: str | Path = DEFAULT_INDEX_DIR, dataset: str = DEFAULT_DATASET, strategy: str = DEFAULT_STRATEGY, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP, embedder: HashingEmbedder | None = None, items: list[SupportFaq] | None = None, save: bool = True) -> tuple[FaissIndex, list[SupportFaq]]` — Собирает (и при ``save=True`` сохраняет) индекс FAQ.
- `load_faq_retriever(index_dir: str | Path = DEFAULT_INDEX_DIR, dataset: str = DEFAULT_DATASET, strategy: str = DEFAULT_STRATEGY, top_k: int = 5, min_score: float = 0.0, root: Path = PROJECT_ROOT) -> RagRetriever` — Загружает индекс FAQ с диска как :class:`RagRetriever`.
- `build_faq_context(sources: list[RagSource], max_chars: int = 6000) -> str` — Собирает контекст FAQ с указанием вопроса и раздела.
- `main() -> int`

### `agent/support_scenarios.py` (355 строк)

День 33: сквозной сценарий ассистента поддержки (офлайн-проверка).

**Функции:**

- `now_iso() -> str`
- `check_faq_index(index_dir: str, strategy: str, top_k: int) -> dict` — Собирает индекс FAQ и проверяет полноту поиска по контрольным вопросам.
- `check_mcp(runtime, ticket_id: str) -> dict` — Проверяет MCP-CRM: пользователь, тикет, список, поиск, готовый контекст.
- `check_qa(assistant: SupportAssistant, ticket_id: str, top_k: int) -> dict` — Проверяет привязку тикета и сквозной ответ по контрольным вопросам.
- `render_report(index_info: dict, mcp_info: dict, qa: dict, all_ok: bool) -> str`
- `main() -> int`

### `agent/task_scenarios.py` (358 строк)

Day 13: состояние задачи как конечный автомат.

**Классы:**

- `FakeClient` — Имитирует LLM: показывает состояние задачи и отвечает на проверку этапа.
  - `complete(self, messages, temperature = None, max_tokens = None) -> Completion`

**Функции:**

- `build_client(config: dict, offline: bool) -> object`
- `new_agent(config: dict, offline: bool, task: TaskStateMachine) -> Agent` — Создаёт агента с конечным автоматом задачи в отдельной временной сессии.
- `show_context(agent: Agent) -> None` — Печатает сообщения, которые реально уйдут в модель (состояние + история).
- `run_scenario(config: dict, offline: bool) -> None` — Прогоняет фиксированный сценарий: автопродвижение + пауза + валидация.
- `run_interactive(config: dict, offline: bool) -> None` — Живой ввод: пользователь сам управляет конечным автоматом задачи.
- `main() -> None`

### `agent/task_state.py` (739 строк)

Состояние задачи как конечный автомат (Finite State Machine).

**Классы:**

- `TransitionVerdict` — Вердикт проверки: перепрыгнул ли ассистент этап задачи.
- `TaskState` — Формализованное состояние задачи: этап, шаг, ожидаемое действие.
  - `begin(cls, description: str, expected_action: str = '') -> 'TaskState'` — Создаёт состояние только что начатой задачи (этап ``planning``).
  - `from_dict(cls, data: dict) -> 'TaskState'` — Восстанавливает состояние из словаря (например, из JSON-сессии).
  - `to_dict(self) -> dict` — Сериализует состояние в словарь для сохранения в JSON.
  - `complete(self) -> bool` — Отмечает текущий этап как выполненный (завершён).
  - `advance(self) -> str` — Переводит задачу на следующий этап (по автомату).
  - `set_stage(self, stage: str) -> str` — Явно переводит задачу на указанный этап, проверяя допустимость.
  - `set_step(self, step: int) -> int` — Устанавливает номер текущего шага (только вперёд, не меньше текущего).
  - `set_expected_action(self, action: str) -> str` — Устанавливает ожидаемое действие.
  - `pause(self) -> bool` — Приостанавливает задачу. Разрешено на любом этапе.
  - `resume(self) -> bool` — Возобновляет задачу с того же места (без повторных объяснений).
  - `is_paused(self) -> bool` — Приостановлена ли задача.
  - `is_done(self) -> bool` — Завершена ли задача.
  - `stage_label(self) -> str` — Человекочитаемое название текущего этапа.
  - `system_message(self) -> str` — System-сообщение с текущим состоянием задачи (для подстановки в запрос).
  - `summarize(self) -> str` — Краткая человекочитаемая сводка состояния задачи.
- `TaskStateMachine` — Конечный автомат состояния задачи.
  - `begin(cls, description: str, expected_action: str = '', auto_advance: bool = False, strict: bool = True) -> 'TaskStateMachine'` — Создаёт автомат для только что начатой задачи.
  - `from_dict(cls, data: dict | None, auto_advance: bool = False, strict: bool = True) -> 'TaskStateMachine'` — Восстанавливает автомат из словаря (None — пустое состояние).
  - `auto_advance(self) -> bool` — Автоматически ли продвигать задачу, когда этап выполнен.
  - `auto_advance(self, value: bool) -> None`
  - `strict(self) -> bool` — Строго ли контролировать переходы (запрещать перепрыгивание этапа).
  - `strict(self, value: bool) -> None`
  - `state(self) -> TaskState` — Текущее состояние задачи.
  - `stage(self) -> str` — Текущий этап задачи.
  - `step(self) -> int` — Текущий шаг задачи.
  - `expected_action(self) -> str` — Ожидаемое действие на текущем шаге.
  - `description(self) -> str` — Описание задачи.
  - `is_paused(self) -> bool` — Приостановлена ли задача.
  - `is_done(self) -> bool` — Завершена ли задача.
  - `is_completed(self) -> bool` — Выполнен ли текущий этап (можно ли переходить дальше).
  - `has_task(self) -> bool` — Есть ли у автомата осмысленная задача (есть описание или этап).
  - `advance(self) -> str` — Переводит задачу на следующий этап по автомату.
  - `complete(self) -> bool` — Отмечает текущий этап выполненным (разрешает переход дальше).
  - `set_stage(self, stage: str) -> str` — Явно переводит задачу на указанный этап (с проверкой перехода).
  - `set_step(self, step: int) -> int` — Устанавливает номер текущего шага.
  - `set_expected_action(self, action: str) -> str` — Устанавливает ожидаемое действие.
  - `pause(self) -> bool` — Приостанавливает задачу на текущем этапе.
  - `resume(self) -> bool` — Возобновляет задачу с того же места.
  - `reset(self, description: str = '', expected_action: str = '') -> None` — Сбрасывает автомат в начальное состояние новой задачи.
  - `system_message(self) -> str` — System-сообщение с текущим состоянием задачи.
  - `completion_check_messages(self, user_request: str, assistant_reply: str, recent_messages: list[dict] | None = None) -> list[dict]` — Сообщения короткой проверки: выполнено ли ожидаемое действие этапа.
  - `guard_request(self, user_request: str) -> str | None` — Проверяет, не пытается ли пользователь перепрыгнуть текущий этап.
  - `transition_guard_messages(self, user_request: str, assistant_reply: str) -> list[dict]` — Сообщения короткой проверки: перепрыгнул ли ассистент этап задачи.
  - `refusal_message(self, reason: str) -> str` — Текст отказа ассистента: объясняет, какой этап перепрыгнут и почему.
  - `summarize(self) -> str` — Краткая сводка состояния задачи.
  - `to_dict(self) -> dict` — Сериализует автомат в словарь для сохранения в JSON.

**Функции:**

- `parse_transition_verdict(text: str) -> 'TransitionVerdict'` — Разбирает ответ модели на проверку перепрыгивания этапа.
- `parse_completion_verdict(text: str) -> bool` — Разбирает ответ модели на проверку завершения этапа.

### `agent/tokens.py` (112 строк)

Подсчёт токенов и учёт стоимости по мере развития диалога агента.

**Классы:**

- `ContextOverflowError` (RuntimeError) — Поднимается, когда диалог превышает контекстный лимит модели.
- `TurnUsage` — Учёт токенов и стоимости одного хода (запрос + ответ модели).
  - `total_request_tokens(self) -> int` — Токены, отправленные в модель: история + текущий запрос.
  - `total_tokens(self) -> int` — Все токены хода: запрос + ответ модели.
- `UsageReport` — Накопительная статистика токенов и стоимости всей сессии.
  - `total_request_tokens(self) -> int`
  - `total_response_tokens(self) -> int`
  - `total_tokens(self) -> int`
  - `total_prompt_tokens(self) -> int | None` — Реальные входные токены за сессию (если API их вернул).
  - `total_completion_tokens(self) -> int | None` — Реальные выходные токены за сессию (если API их вернул).
  - `total_cost(self) -> float | None`
  - `record(self, request_tokens: int, history_tokens: int, response_tokens: int | None, prompt_tokens: int | None, completion_tokens: int | None) -> TurnUsage` — Добавляет ход в отчёт и считает его стоимость по тарифам.

**Функции:**

- `estimate_tokens(text: str) -> int` — Грубая оценка числа токенов в тексте (~4 символа на токен).
- `estimate_messages_tokens(messages: list[dict]) -> int` — Сумма токенов по всем сообщениям истории диалога.

### `agent/tool_scenarios.py` (150 строк)

Day 17: первый MCP-инструмент — публичные проекты GitFlic.

**Функции:**

- `main() -> None`

### `agent/transition_scenarios.py` (439 строк)

Day 15: контролируемые переходы состояний задачи.

**Классы:**

- `FakeClient` — Имитирует LLM: показывает состояние задачи и отвечает на проверки.
  - `make_jump(self) -> None` — Включает режим, в котором обычный ответ ассистента «перепрыгивает» этап.
  - `complete(self, messages, temperature = None, max_tokens = None) -> Completion`

**Функции:**

- `build_client(config: dict, offline: bool) -> object`
- `new_agent(config: dict, offline: bool, task: TaskStateMachine) -> Agent` — Создаёт агента с контролируемым жизненным циклом задачи.
- `show_context(agent: Agent) -> None` — Печатает сообщения, которые реально уйдут в модель (состояние + история).
- `print_states() -> None` — Печатает допустимые состояния и разрешённые переходы автомата.
- `run_scenario(config: dict, offline: bool) -> None` — Прогоняет фиксированный сценарий контролируемых переходов.
- `run_interactive(config: dict, offline: bool) -> None` — Живой ввод: пользователь сам управляет контролируемым жизненным циклом.
- `main() -> None`

### `agent/web_chat.py` (1362 строк)

Day 27–28: веб-чат с локальной и облачной LLM.

**Классы:**

- `RagUnavailableError` (RuntimeError) — Режим «рецепты» выбран, но RAG-индекс недоступен.
- `ModelSourceUnavailableError` (RuntimeError) — Выбран источник модели («local»/«cloud»), который недоступен.
- `StubLLMClient` — Офлайн-клиент локальной LLM, совместимый с :class:`agent.agent.Agent`.
  - `complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Completion`
- `ChatReply` — Нормализованный ответ ассистента (без RAG или с найденными источниками).
- `ChatChannel` — Один канал генерации: обычный агент и (опционально) RAG-система.
- `ChatSession` — Одна сессия веб-чата: диалог с моделью и его история на диске.
  - `session_id(self) -> str`
  - `messages(self) -> list[dict]` — История сообщений сессии (user/assistant).
  - `rag_enabled(self) -> bool` — Доступен ли RAG хотя бы в одном канале.
  - `sources(self) -> list[str]` — Доступные источники модели (``local``/``cloud``).
  - `ask(self, message: str, use_rag: bool = False, model_source: str = 'local', temperature: float | None = None, max_tokens: int | None = None, context_window: int | None = None, system_prompt: str | None = None) -> ChatReply` — Отправляет сообщение в выбранную модель и возвращает ответ.
  - `reset(self) -> None` — Очищает историю диалога сессии.
- `WebChatBackend` — Бэкенд веб-чата: локальная и облачная модели + реестр сессий диалога.
  - `mode(self) -> str` — Режим работы: ``offline`` (стаб) или ``live`` (llama-server).
  - `ready(self) -> bool` — Готов ли бэкенд принимать запросы.
  - `rag_enabled(self) -> bool` — Доступен ли RAG-режим (индекс рецептов загружен).
  - `rag_error(self) -> str | None` — Причина недоступности RAG (если индекс не загрузился).
  - `model_label(self) -> str`
  - `cloud_enabled(self) -> bool` — Доступен ли облачный канал (клиент настроен).
  - `cloud_model_label(self) -> str` — Метка облачной модели из конфигурации (пусто, если не задана).
  - `model_sources(self) -> list[str]` — Доступные источники модели в порядке отображения.
  - `variants(self) -> list[ModelVariant]` — Доступные варианты локальной модели (квантования).
  - `presets(self) -> dict[str, GenerationParams]` — Именованные пресеты параметров генерации.
  - `prompt_templates(self) -> dict` — Доступные prompt-шаблоны (имя → :class:`PromptTemplate`).
  - `local_variant_label(self) -> str` — Метка выбранного варианта локальной модели (квантования).
  - `prompt_template_label(self) -> str` — Имя активного prompt-шаблона.
  - `generation_params(self) -> GenerationParams` — Текущие параметры генерации по умолчанию.
  - `ctx_size(self) -> int` — Размер контекста llama-server (``-c``).
  - `endpoint(self) -> str` — Адрес модели, к которому обращается веб-чат.
  - `start(self) -> WebChatBackend` — Готовит модели: поднимает llama-server, стаб и/или облачный клиент.
  - `stop(self) -> None` — Останавливает локальный сервер, если он был запущен.
  - `status(self) -> dict` — Сводка состояния бэкенда для ``GET /api/status``.
  - `configure(self, ctx_size: int | None = None, variant: str | None = None) -> dict` — Применяет настройки уровня сервера (День 29).
  - `chat(self, message: str, session_id: str | None = None, use_rag: bool = False, model_source: str = 'local', temperature: float | None = None, max_tokens: int | None = None, context_window: int | None = None, preset: str | None = None, prompt_template: str | None = None) -> dict` — Отправляет сообщение в выбранную модель и возвращает ответ.
  - `reset(self, session_id: str | None) -> str` — Сбрасывает историю сессии и возвращает её id.
  - `history(self, session_id: str | None) -> list[dict]` — История сообщений сессии.

**Функции:**

- `create_app(backend: WebChatBackend) -> Flask` — Создаёт Flask-приложение поверх готового :class:`WebChatBackend`.

### `agent/web_chat_scenarios.py` (263 строк)

Day 27: сквозная проверка веб-чата с локальной и облачной LLM.

**Классы:**

- `CheckResult` — Результат одной проверки сценария.

**Функции:**

- `main() -> int`


## Пакет `mcp/`

### `mcp/fs_server.py` (574 строк)

День 34: MCP-сервер файловых операций (чтение, поиск, анализ, запись).

**Функции:**

- `workspace_root() -> Path` — Корень рабочей области (можно переопределить через env ``FS_ROOT``).
- `list_files(path: Annotated[str, Field(description='Префикс пути/каталог (пусто — вся область).')] = '', pattern: Annotated[str, Field(description="Glob-шаблон имени файла, например '*.py'.")] = '', limit: Annotated[int, Field(description='Сколько файлов вернуть (по умолчанию 200).')] = 200) -> dict` — Перечислить файлы рабочей области.
- `read_file(path: Annotated[str, Field(description='Путь к файлу относительно корня области.')], start_line: Annotated[int, Field(description='Первая строка (1-based, по умолчанию 1).')] = 1, end_line: Annotated[int, Field(description='Последняя строка (0 — до конца файла).')] = 0, max_chars: Annotated[int, Field(description='Максимум символов в ответе.')] = 8000) -> dict` — Прочитать файл или диапазон строк.
- `search_files(query: Annotated[str, Field(description='Строка или регулярное выражение для поиска.')], path: Annotated[str, Field(description='Каталог/префикс для поиска (пусто — вся область).')] = '', glob: Annotated[str, Field(description="Glob-шаблон файлов, например '*.py'.")] = '', regex: Annotated[bool, Field(description='Считать query регулярным выражением.')] = False, case_sensitive: Annotated[bool, Field(description='Учитывать регистр.')] = False, limit: Annotated[int, Field(description='Сколько файлов-совпадений вернуть.')] = 100, context: Annotated[int, Field(description='Сколько соседних строк показывать (0 — только строка).')] = 0) -> dict` — Найти текст по нескольким файлам.
- `analyze_file(path: Annotated[str, Field(description='Путь к файлу для анализа.')]) -> dict` — Проанализировать структуру файла.
- `file_info(path: Annotated[str, Field(description='Путь к файлу.')]) -> dict` — Метаданные файла: размер, время изменения, число строк.
- `write_file(path: Annotated[str, Field(description='Путь к файлу для записи.')], content: Annotated[str, Field(description='Полное содержимое файла.')], overwrite: Annotated[bool, Field(description='Разрешить перезапись существующего файла.')] = True, create_dirs: Annotated[bool, Field(description='Создавать недостающие каталоги.')] = True) -> dict` — Создать или перезаписать файл.
- `edit_file(path: Annotated[str, Field(description='Путь к изменяемому файлу.')], old: Annotated[str, Field(description='Подстрока, которую нужно заменить.')], new: Annotated[str, Field(description='Новая подстрока.')], count: Annotated[int, Field(description='Сколько вхождений заменить (0 — все).')] = 1) -> dict` — Заменить подстроку в файле.
- `make_diff(path: Annotated[str, Field(description='Файл или каталог (пусто — вся область).')] = '', baseline_root: Annotated[str, Field(description='Каталог-эталон для сравнения (difflib). Пусто — git diff.')] = '', max_chars: Annotated[int, Field(description='Максимум символов diff.')] = 8000) -> dict` — Показать изменения файлов.
- `main() -> None`

### `mcp/gitflic_api.py` (128 строк)

Day 18: общий слой доступа к публичному API GitFlic.

**Функции:**

- `api_url() -> str` — Базовый адрес API GitFlic (можно переопределить через окружение).
- `token() -> str` — Токен доступа GitFlic из переменной окружения (может быть пустым).
- `request_timeout() -> float` — Таймаут одного HTTP-запроса к GitFlic (сек), с учётом ``GITFLIC_TIMEOUT``.
- `request_retries() -> int` — Сколько раз повторять запрос при таймауте/сетевой ошибке (``GITFLIC_RETRIES``).
- `request_json(path: str, params: dict | None = None) -> dict` — Выполняет GET-запрос к API GitFlic и возвращает разобранный JSON.
- `normalize_project(raw: dict) -> dict` — Приводит проект GitFlic к компактному, удобному агенту виду.
- `fetch_public_projects(query: str = '', page: int = 0, size: int = 100) -> dict` — Возвращает одну страницу публичных проектов GitFlic и данные пагинации.

### `mcp/gitflic_cron.py` (209 строк)

Day 18: запуск подсчёта публичных проектов GitFlic по cron.

**Функции:**

- `make_scheduler(store: str | None = None, jobs: str | None = None) -> ProjectCountScheduler`
- `run_once(query: str = '', max_pages: int = 0, store: str | None = None) -> dict` — Один подсчёт с сохранением снимка; возвращает агрегированный результат.
- `crontab_lines() -> list[str]` — Готовые строки crontab для периодического подсчёта и выполнения задач.
- `main() -> None`

### `mcp/gitflic_scheduler.py` (446 строк)

Day 18: планировщик и фоновые задачи вокруг публичного API GitFlic.

**Классы:**

- `CountStore` (_JsonStore) — JSON-хранилище снимков подсчёта публичных проектов.
  - `load(self) -> dict`
  - `append(self, snapshot: dict) -> dict`
  - `latest(self) -> dict | None`
  - `history(self, limit: int = 20) -> list[dict]`
- `JobStore` (_JsonStore) — JSON-хранилище расписаний планировщика (задач подсчёта).
  - `load(self) -> dict`
  - `put(self, job: dict) -> None`
  - `get(self, job_id: str) -> dict | None`
  - `all(self) -> list[dict]`
- `ProjectCounter` — Считает публичные проекты GitFlic и сохраняет снимок в JSON.
  - `count_now(self, query: str = '', job_id: str | None = None, trigger: str = 'manual') -> dict` — Делает подсчёт, сохраняет снимок и возвращает агрегированный результат.
- `ProjectCountScheduler` — Планировщик отложенных и периодических подсчётов публичных проектов.
  - `counter(self) -> ProjectCounter`
  - `schedule(self, query: str = '', delay_seconds: float = 0.0, interval_seconds: float | None = None, run_immediately: bool = False) -> dict` — Ставит задачу подсчёта, сохраняет её и возвращает описание.
  - `due_jobs(self, at: datetime | None = None) -> list[dict]` — Задачи, готовые к выполнению (``nextRunAt <= now`` и не отменённые).
  - `run_due(self, at: datetime | None = None) -> list[dict]` — Выполняет все готовые задачи и возвращает результаты их запусков.
  - `list_jobs(self) -> list[dict]`
  - `cancel(self, job_id: str) -> bool`
  - `start_daemon(self) -> None` — Запускает поток, который сам выполняет готовые задачи раз в секунду.
  - `stop_daemon(self) -> None`

**Функции:**

- `default_max_pages() -> int` — Сколько страниц обходить по умолчанию (с учётом ``GITFLIC_MAX_PAGES``).
- `now_iso() -> str` — Текущее время в ISO-8601 с локальным часовым поясом.
- `aggregate_projects(projects: list[dict]) -> dict` — Агрегирует список проектов: итог, языки, владельцы, топики.

### `mcp/gitflic_server.py` (270 строк)

Day 17–18: собственный MCP-сервер вокруг публичного API GitFlic.

**Функции:**

- `get_public_projects(query: Annotated[str, Field(description='Необязательная подстрока названия проекта для поиска.')] = '', page: Annotated[int, Field(description='Номер страницы, начиная с 0 (первая страница — 0).')] = 0, size: Annotated[int, Field(description='Сколько проектов вернуть на странице (1..100).')] = 10) -> dict` — Получить список публичных проектов GitFlic.
- `count_public_projects(query: Annotated[str, Field(description='Необязательная подстрока названия проекта для поиска.')] = '', max_pages: Annotated[int, Field(description='Сколько страниц обойти максимум (0 — по умолчанию 5 или значение переменной окружения GITFLIC_MAX_PAGES).')] = 0) -> dict` — Посчитать количество публичных проектов GitFlic и сохранить данные в JSON.
- `schedule_project_count(query: Annotated[str, Field(description='Необязательная подстрока названия проекта для поиска.')] = '', delay_seconds: Annotated[float, Field(description='Задержка перед первым подсчётом, секунды.')] = 0.0, interval_seconds: Annotated[float | None, Field(description='Период повторения в секундах (None — один отложенный запуск).')] = None, run_immediately: Annotated[bool, Field(description='Выполнить первый подсчёт сразу, не дожидаясь задержки.')] = False) -> dict` — Запустить подсчёт публичных проектов по расписанию (в фоне).
- `list_count_jobs() -> dict` — Показать фоновые задачи подсчёта: статус, следующий запуск, число прогонов.
- `cancel_count_job(job_id: Annotated[str, Field(description='Идентификатор задачи из list_count_jobs.')]) -> dict` — Отменить фоновую периодическую задачу подсчёта по её jobId.
- `get_project_count_report(history_limit: Annotated[int, Field(description='Сколько последних снимков вернуть в истории.')] = 20) -> dict` — Вернуть агрегированный результат подсчёта и историю сохранённых снимков.
- `main() -> None`

### `mcp/pipeline_server.py` (405 строк)

Day 19: MCP-сервер с композицией инструментов (search → summarize → saveToFile).

**Функции:**

- `now_iso() -> str` — Текущее время в ISO-8601 с локальным часовым поясом.
- `search(query: Annotated[str, Field(description='Подстрока названия проекта для поиска (может быть пустой).')] = '', page: Annotated[int, Field(description='Номер страницы, начиная с 0 (первая страница — 0).')] = 0, size: Annotated[int, Field(description='Сколько проектов вернуть на странице (1..100).')] = 10) -> dict` — Получить публичные проекты GitFlic — первый шаг пайплайна.
- `summarize(items: Annotated[list[dict], Field(description='Проекты от search: поля title, description, language, owner, topics.')] = None, title: Annotated[str, Field(description='Заголовок/тема отчёта, подмешивается в текст выжимки.')] = '', max_sentences: Annotated[int, Field(description='Сколько лучших предложений оставить в выжимке.')] = 3, max_keywords: Annotated[int, Field(description='Сколько ключевых слов вернуть.')] = 8) -> dict` — Сделать выжимку из проектов — второй шаг пайплайна.
- `save_to_file(content: Annotated[str | dict | list, Field(description='Что сохранить: текст, объект или список (для format=json).')] = '', path: Annotated[str, Field(description='Путь к файлу внутри проекта (по умолчанию history/pipeline_result.md).')] = DEFAULT_OUTPUT, format: Annotated[str, Field(description='Формат: markdown | text | json.')] = 'markdown', append: Annotated[bool, Field(description='Дописать в конец файла, а не перезаписать.')] = False) -> dict` — Сохранить результат — третий шаг пайплайна.
- `run_pipeline(query: Annotated[str, Field(description='Поисковый запрос для первого шага (search).')] = '', output_path: Annotated[str, Field(description='Куда сохранить отчёт (saveToFile).')] = DEFAULT_OUTPUT, size: Annotated[int, Field(description='Сколько проектов запросить у GitFlic.')] = 10, max_sentences: Annotated[int, Field(description='Сколько предложений оставить в выжимке.')] = 3) -> dict` — Выполнить всю цепочку search → summarize → saveToFile одним вызовом.
- `main() -> None`

### `mcp/pipeline_text.py` (182 строк)

Day 19: локальная экстрактивная суммаризация для MCP-инструмента ``summarize``.

**Функции:**

- `tokenize(text: str) -> list[str]` — Слова текста в нижнем регистре без стоп-слов и коротких токенов.
- `split_sentences(text: str) -> list[str]` — Делит текст на предложения и отбрасывает пустые.
- `item_to_text(item: dict) -> str` — Текст проекта для суммаризации: название, описание, язык, владелец, топики.
- `item_bullet(item: dict, max_topics: int = 4) -> str` — Краткий пункт по проекту: название, язык, владелец, топики.
- `top_keywords(text: str, limit: int = 8) -> list[str]` — Ключевые слова: самые частые значимые слова текста.
- `summarize_text(text: str, max_sentences: int = 3, max_keywords: int = 8) -> dict` — Экстрактивная выжимка из текста: лучшие предложения и ключевые слова.
- `summarize_items(items: list[dict], title: str = '', max_sentences: int = 3, max_keywords: int = 8, max_bullets: int = 10) -> dict` — Суммаризует список проектов GitFlic: выжимка, пункты, ключевые слова.

### `mcp/project_server.py` (322 строк)

День 31: MCP-сервер, подключающий ассистента к самому проекту.

**Функции:**

- `get_current_branch() -> dict` — Получить текущую git-ветку проекта.
- `get_repo_status(limit: Annotated[int, Field(description='Сколько файлов показывать в каждом списке (по умолчанию 30).')] = 30) -> dict` — Получить состояние рабочего дерева проекта.
- `list_project_files(path: Annotated[str, Field(description="Префикс пути для фильтра (например, 'agent/' или 'docs').")] = '', limit: Annotated[int, Field(description='Сколько файлов вернуть (по умолчанию 200).')] = 200) -> dict` — Перечислить файлы проекта под контролем версий.
- `get_recent_commits(limit: Annotated[int, Field(description='Сколько последних коммитов вернуть (по умолчанию 10).')] = 10) -> dict` — Получить последние коммиты репозитория.
- `get_diff(base: Annotated[str, Field(description="База сравнения (по умолчанию 'HEAD'); пусто — рабочие изменения.")] = 'HEAD', staged: Annotated[bool, Field(description='Показать изменения индекса (--staged), а не рабочего дерева.')] = False, path: Annotated[str, Field(description='Ограничить diff конкретным путём (пусто — весь проект).')] = '', max_chars: Annotated[int, Field(description='Максимальная длина diff в символах (по умолчанию 4000).')] = 4000) -> dict` — Получить diff изменений проекта.
- `main() -> None`

### `mcp/report_server.py` (314 строк)

Day 20: третий MCP-сервер оркестрации — журнал и индекс отчётов.

**Функции:**

- `now_iso() -> str` — Текущее время в ISO-8601 с локальным часовым поясом.
- `list_reports(directory: Annotated[str, Field(description='Каталог с отчётами внутри проекта.')] = DEFAULT_REPORTS_DIR) -> dict` — Перечислить сохранённые отчёты в каталоге проекта.
- `record_run(query: Annotated[str, Field(description='Запрос/тема запуска флоу.')] = '', report_path: Annotated[str, Field(description='Путь сохранённого отчёта внутри проекта.')] = '', total_projects: Annotated[int, Field(description='Сколько проектов было в отчёте (0 — неизвестно).')] = 0, tags: Annotated[list[str] | None, Field(description='Метки запуска (необязательно).')] = None) -> dict` — Записать запуск флоу в JSON-журнал.
- `build_report_index(directory: Annotated[str, Field(description='Каталог с отчётами внутри проекта.')] = DEFAULT_REPORTS_DIR, path: Annotated[str, Field(description='Куда сохранить индекс внутри проекта.')] = DEFAULT_INDEX) -> dict` — Собрать единый Markdown-индекс отчётов и журнала запусков.
- `main() -> None`

### `mcp/review_server.py` (346 строк)

День 32: MCP-сервер, отдающий ассистенту данные для ревью PR.

**Функции:**

- `repo_root() -> Path` — Корень проверяемого репозитория (можно переопределить через env).
- `get_pr_diff(base: Annotated[str, Field(description="База сравнения (SHA/ветка), по умолчанию 'HEAD'.")] = 'HEAD', head: Annotated[str, Field(description='Верхняя ревизия (SHA/ветка); пусто — рабочее дерево.')] = '', path: Annotated[str, Field(description='Ограничить diff конкретным путём (пусто — весь PR).')] = '', max_chars: Annotated[int, Field(description='Максимальная длина diff в символах (по умолчанию 20000).')] = 20000) -> dict` — Получить diff pull request (base → head).
- `list_changed_files(base: Annotated[str, Field(description="База сравнения (SHA/ветка), по умолчанию 'HEAD'.")] = 'HEAD', head: Annotated[str, Field(description='Верхняя ревизия (SHA/ветка); пусто — рабочее дерево.')] = '') -> dict` — Перечислить изменённые в PR файлы со статусом и числом добавленных строк.
- `get_file_content(path: Annotated[str, Field(description='Путь к файлу относительно корня репозитория.')], ref: Annotated[str, Field(description='Ревизия (SHA/ветка); пусто — рабочий файл.')] = '', max_chars: Annotated[int, Field(description='Максимальная длина содержимого в символах (по умолчанию 20000).')] = 20000) -> dict` — Прочитать содержимое файла на выбранной ревизии.
- `get_pr_metadata() -> dict` — Получить метаданные pull request из GitHub-события.
- `main() -> None`

### `mcp/support_server.py` (374 строк)

День 33: MCP-сервер поддержки — CRM с пользователями и тикетами.

**Функции:**

- `data_dir() -> Path` — Каталог с JSON-выгрузкой CRM (можно переопределить через env).
- `render_ticket_context(ticket: dict, user: dict | None = None) -> str` — Человекочитаемый контекст «пользователь + тикет + переписка».
- `get_user(user_id: Annotated[int, Field(description='Идентификатор пользователя (id).')]) -> dict` — Получить профиль пользователя по идентификатору.
- `find_user(query: Annotated[str, Field(description='Подстрока для поиска: логин, e-mail или имя.')], limit: Annotated[int, Field(description='Сколько результатов вернуть (по умолчанию 5).')] = 5) -> dict` — Найти пользователя по логину, e-mail или имени.
- `get_ticket(ticket_id: Annotated[str, Field(description="Идентификатор тикета, например 'T-1024'.")]) -> dict` — Получить полный тикет по идентификатору (включая переписку).
- `list_tickets(user_id: Annotated[str, Field(description='Фильтр по id пользователя (пусто — все).')] = '', status: Annotated[str, Field(description='Фильтр по статусу (пусто — все).')] = '', limit: Annotated[int, Field(description='Сколько тикетов вернуть (по умолчанию 20).')] = 20) -> dict` — Перечислить тикеты (краткая форма) с фильтрами по пользователю и статусу.
- `search_tickets(query: Annotated[str, Field(description='Текст для поиска по теме, тегам и переписке.')], limit: Annotated[int, Field(description='Сколько результатов вернуть (по умолчанию 10).')] = 10, scope: Annotated[str, Field(description="Где искать: 'all' (тема/теги/категория/переписка) или 'meta' (без переписки).")] = 'all') -> dict` — Найти тикеты по тексту: тема, теги и (при scope='all') переписка.
- `get_ticket_context(ticket_id: Annotated[str, Field(description="Идентификатор тикета, например 'T-1024'.")]) -> dict` — Собрать готовый контекст тикета: пользователь + тикет + переписка.
- `main() -> None`
