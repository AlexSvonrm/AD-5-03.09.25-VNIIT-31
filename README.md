[![Main Kittygram workflow](https://github.com/AlexSvonrm/kittygram_final/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/AlexSvonrm/kittygram_final/actions/workflows/main.yml)

# Kittygram

Kittygram — Социальная сеть для любителей котиков 🐱. 

## О проекте

Kittygram - это полнофункциональная социальная сеть для владельцев котиков, где пользователи могут делиться фотографиями своих питомцев, отмечать достижения и общаться с другими любителями кошек.
Этапы работы позволяют:
- настроить запуск проекта Kittygram в контейнерах;
- настроить автоматическое тестирование и деплой (CI/CD) этого проекта на удалённый сервер

### Основные функции:

- **📱 Регистрация и аутентификация** - система пользователей с JWT токенами
- **🐱 Профили котиков** - создание и редактирование профилей питомцев
- **🏆 Достижения** - система достижений и наград для котиков
- **📸 Медиа-контент** - загрузка фотографий котиков
- **👥 Социальные функции** - просмотр профилей других пользователей

## Технологический стек

### Backend:
- **Python 3.11** - основной язык программирования
- **Django 4.2+** - веб-фреймворк
- **Django REST Framework** - REST API
- **Djoser** - аутентификация и управление пользователями
- **PostgreSQL** - база данных
- **Gunicorn** - WSGI сервер
- **Pillow** - работа с изображениями

### Frontend:
- **React** - пользовательский интерфейс
- **Modern CSS** - стилизация
- **Static files** - статические файлы

### Infrastructure:
- **Docker** - контейнеризация
- **Docker Compose** - оркестрация контейнеров
- **Nginx** - веб-сервер и прокси
- **PostgreSQL** - СУБД

## Установка и настройка

### 1. Клонирование репозитория и подготовка 

1. Клонируйте репозиторий на свой компьютер:

    ```bash
    git clone https://github.com/AlexSvonrm/kittygram_final
    cd kittygram_fina
    ```
 Cоздать и активировать виртуальное окружение:

    ```bash
    python -m venv venv
    source venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
2. Создайте файл .env и заполните его своими данными на основани примера в файле .env.example.


### Создание Docker-образов

1.  Список команд для создания образов, (username заменить на ваш логин на DockerHub):

    ```bash
    cd frontend
    docker build -t username/kittygram_frontend .
    cd ../backend
    docker build -t username/kittygram_backend .
    cd ../nginx
    docker build -t username/kittygram_gateway . 
    ```

2. Загрузите образы на DockerHub:

    ```bash
    docker push username/kittygram_frontend
    docker push username/kittygram_backend
    docker push username/kittygram_gateway
    ```

### Деплой на сервере

1. Подключитесь к удаленному серверу

    ```bash
    ssh -i путь_до_файла_с_SSH_ключом/название_файла_с_SSH_ключом имя_пользователя@ip_адрес_сервера 
    ```

2. Создайте на сервере директорию kittygram

    ```bash
    mkdir kittygram
    ```

3. Установка docker compose на сервер:

    ```bash
    sudo apt update
    sudo apt install curl
    curl -fSL https://get.docker.com -o get-docker.sh
    sudo sh ./get-docker.sh
    sudo apt-get install docker-compose-plugin
    ```

4. В директорию kittygram/ скопируйте файлы docker-compose.production.yml и подготовленный .env:

    ```bash
    scp -i path_to_SSH/SSH_name docker-compose.production.yml username@server_ip:/home/username/kittygram/docker-compose.production.yml
    * path_to_SSH — путь к файлу с SSH-ключом;
    * SSH_name — имя файла с SSH-ключом (без расширения);
    * username — ваше имя пользователя на сервере;
    * server_ip — IP вашего сервера.
    ```

5. Запустите docker compose в режиме демона:

    ```bash
    sudo docker compose -f docker-compose.production.yml up -d
    ```

6. Выполните миграции, соберите статические файлы бэкенда и скопируйте их в /backend_static/static/:

    ```bash
    sudo docker compose -f docker-compose.production.yml exec backend python manage.py migrate
    sudo docker compose -f docker-compose.production.yml exec backend python manage.py collectstatic
    sudo docker compose -f docker-compose.production.yml exec backend cp -r /app/collected_static/. /backend_static/static/
    ```

7. На сервере в редакторе nano откройте конфиг Nginx:

    ```bash
    sudo nano /etc/nginx/sites-enabled/default
    ```

8. Измените настройки location в секции server:

    ```bash
    location / {
        proxy_set_header Host $http_host;
        proxy_pass http://127.0.0.1:9000;
    }
    ```

9. Проверьте работоспособность конфига Nginx:

    ```bash
    sudo nginx -t
    ```
    
10. Перезапускаем Nginx
    ```bash
    sudo service nginx reload
    ```

### Настройка CI/CD

1. Файл workflow уже написан. Он находится в директории

    ```bash
    kittygram/.github/workflows/main.yml
    ```

2. Для адаптации его на своем сервере добавьте секреты в GitHub Actions:

    ```bash
    DOCKER_USERNAME                # имя пользователя в DockerHub
    DOCKER_PASSWORD                # пароль пользователя в DockerHub
    HOST                           # ip адресс сервера
    USER                           # имя пользователя на сервере
    SSH_KEY                        # закрытый ssh-ключ
    SSH_PASSPHRASE                 # пароль для ssh-ключа

    TELEGRAM_TO                    # id телеграм-аккаунта (применить @userinfobot)
    TELEGRAM_TOKEN                 # токен бота (применить @BotFather)
    ```


### Команда
Исполнитель
* Смирнов Алексей
Контроль
* Игорь Шкода