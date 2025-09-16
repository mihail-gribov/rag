# Настройка Docker окружения

- Номер: 009
- Depends on: [008]
- Спецификация: `.specification/architecture.md#Деплой`
- Тип: infra
- Приоритет: P2
- Статус: todo

## Definition of Ready
- Все зависимости закрыты; ссылки на спецификацию валидны; Inputs/Outputs описаны; команды запуска/проверки даны.

## Контекст
- Необходимо настроить Docker окружение для контейнеризации системы
- Документы: `.specification/architecture.md` - раздел "Деплой" (строки 1196-1383)

## Цель
Создать Docker конфигурацию для развертывания arXiv RAG системы с ChromaDB.

## Inputs
- Спецификация проекта: `.specification/architecture.md` - раздел "Деплой"
- Полная система: все компоненты созданы в предыдущих задачах

## Outputs
- Файл `Dockerfile` для основного приложения
- Файл `docker-compose.yml` для оркестрации сервисов
- Конфигурация для ChromaDB контейнера
- Скрипты для управления Docker окружением

## Объем работ (Scope)
1) Создать Dockerfile для основного приложения
2) Создать docker-compose.yml с сервисами arxiv-rag и chromadb
3) Настроить volumes для персистентного хранения данных
4) Добавить health checks для сервисов
5) Создать скрипты для управления окружением
6) Добавить документацию по использованию Docker
- Out of scope: CI/CD пайплайны (не предусмотрены в спецификации)

## Acceptance criteria
- Команда: `docker-compose build` → успешно собирает образы
- Команда: `docker-compose up -d` → запускает сервисы
- Команда: `docker-compose exec arxiv-rag python cli.py --help` → выводит справку
- ChromaDB доступен на порту 8000

## Тесты
- Unit: тесты сборки Docker образов
- Integration: тесты запуска сервисов
- E2E: тесты работы системы в Docker

## Команды для проверки
```bash
# сборка образов
docker-compose build
# запуск сервисов
docker-compose up -d
# проверка статуса
docker-compose ps
# проверка работы CLI
docker-compose exec arxiv-rag python cli.py --help
# проверка ChromaDB
curl http://localhost:8000/api/v1/heartbeat
```

## Definition of Done
- Lint/type-check/tests/build OK
- Docker окружение настроено согласно спецификации
- Все сервисы запускаются корректно
- Документация по Docker создана
- Статус задачи обновлён на `done`

## Риски и допущения
- Предполагается наличие Docker и Docker Compose в системе
- Предполагается доступ к интернету для загрузки базовых образов
- Предполагается корректная работа всех компонентов системы

## Ссылка на следующую задачу
- 010

