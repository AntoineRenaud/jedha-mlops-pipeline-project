docker run -it `
 -v "${pwd}:/home/app" `
 -e PORT=4000 `
 -e APP_URI=$env:APP_URI `
 -e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID `
 -e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY `
 -e BACKEND_STORE_URI=$env:BACKEND_STORE_URI `
 -e ARTIFACT_STORE_URI=$env:ARTIFACT_STORE_URI `
 -e DATASET_BUCKET=$env:DATASET_BUCKET `
 fraud-detection-model-training pytest tests/ml_test.py