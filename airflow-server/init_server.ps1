mkdir "dags", "logs", "plugins"
docker-compose build
docker-compose up airflow-init
docker-compose up