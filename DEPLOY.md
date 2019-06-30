# Deploy notes

## Heroku

```
heroku login
heroku create boston-housing-svc
git push heroku master
heroku open
```

The app uses a `Procfile` with gunicorn. `runtime.txt` pins Python 3.7.3,
which is on Heroku's supported list.

A trained model (`models/model.pkl`) must be present when the dyno boots,
otherwise the import-time `_load()` will fail. There are two options:

1. Run `python -m src.train` locally and commit the resulting `.pkl` files.
   This is simple but bloats the repo.
2. Add a release phase to `Procfile`:
   ```
   release: python -m src.train
   web: gunicorn app:app --log-file=-
   ```
   The release dyno trains a fresh model on each deploy.

We use option 2 to keep the repo clean.

## Docker

```
docker build -t boston-housing .
docker run -p 5000:5000 boston-housing
```

The image trains a model during `docker build`, so the container starts up
ready to serve.

## Smoke test

After the service is up, hit it:

```
curl -s -XPOST http://localhost:5000/predict \
    -H 'Content-Type: application/json' \
    -d '{"CRIM":0.1,"ZN":0,"INDUS":5,"CHAS":0,"NOX":0.5,"RM":6,
         "AGE":50,"DIS":4,"RAD":1,"TAX":296,"PTRATIO":15.3,
         "B":396.9,"LSTAT":5}'
```

Expected response:

```
{"medv": 24.3}
```
