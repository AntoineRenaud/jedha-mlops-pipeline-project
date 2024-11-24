docker run -it -p 4000:4000 `
    -e PORT=4000 `
    -e APP_URI=$env:APP_URI `
    -e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID `
    -e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY `
    -e ARTIFACT_STORE_URI=$env:ARTIFACT_STORE_URI `
    fraud-api