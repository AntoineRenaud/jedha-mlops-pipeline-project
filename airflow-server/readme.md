add a .env file containing those creds :

AIRFLOW_UID=50000

DATABASE_URL=''
DATABASE=''
USER=''
PASSWORD=''
TRANSACTION_API_URL=''
PREDICTION_API_URL=''


  # In order to add custom dependencies or upgrade provider packages you can use your extended image.
  # Comment the image line, place your Dockerfile in the directory where you placed the docker-compose.yaml
  # and uncomment the "build" line below, Then run `docker-compose build` to build the images.