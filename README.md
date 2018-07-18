# Profile classification project
A model and API for classifying profiles as brands, influencers, or news and media entities.

## Findings
* The training process is described in [this notebook](profile_type_training.ipynb).
* Using the provided data, I achieved 86.5% classification accuracy on a held-out test set, with features extracted from the profile biographies alone.
* The training approach taken was fairly simple (i.e. 'bag of words' and linear classifier), and only took a few minutes of computation on my stock laptop.

## Using the API
The profile classification model was deployed to a corresponding API ([source code and documentation available here](https://github.com/Waldo000000/profile_type_api)). 


The endpoint is available on Heroku here:
```
https://ancient-castle-15317.herokuapp.com/profile
```
It can be invoked by posting profile information to the endpoint, for example, as follows:
```
curl -X POST \
  https://ancient-castle-15317.herokuapp.com/profile \
  -H 'Content-Type: application/json' \
  -d '[
    {
      "name": "SpaceX",
      "bio": "SpaceX designs, manufactures and launches the world’s most advanced rockets and spacecraft.",
      "follower_count": 6500000
    }
]'
```

## Using the trained model
The trained model is available for [download via Google Drive here](https://drive.google.com/file/d/1EWrjN9o3F53An2jmP62Xctt1Zdl5vnYu/view).

Once downloaded, the model can be loaded and used for estimation as follows, by providing a profile biography (note: requires Python 3):

```python
from sklearn.externals import joblib
from pandas import DataFrame

clf = joblib.load('./clf.pkl')

bios = [
  'Keeping you informed about events in your city'
]

DataFrame(
    loadedClf.predict_proba(bios),
    columns=loadedClf.classes_
)
```

This will output the classification probabilities accordingly:
```python
brand   influencer  news and media
0.16789 0.016557    0.815553
```

## Future steps
Following are some ideas for where I’d like to take the project. 

* Modelling/Data exploration
    * Provide more detailed analysis of precision and recall scores for each category (not just mean classification accuracy), eg confusion matrices.
    * As mentioned, I decided to focus on the biography features initially. I'd like to see if adding follower count and/or name are complementary and can improve accuracy.
* API/practical issues
    * API authorization (e.g. require an API key is provided)
    * API request validation (eg with http://flask-marshmallow.readthedocs.io/)
    * API unit/integration tests and deployment pipeline
    * Possibly wrap up the logic in the model training scripts (which were shown in a Jupyter notebook for this demonstration) into reusable libraries for future training of other models, scoring on future validation sets, etc
* Training/Performance
    * Model is currently just over 100M right now. Might be nice to get that down a bit.
    * The model incorporates a calibration step to ensure we can output classification probabilities. Unfortunately, due to https://github.com/scikit-learn/scikit-learn/issues/8710, 
      the calibration training stage is inefficient, as its cross validation folds are nested within those from the hyperparameter grid search. 
      Ideally, grid search would be completed first, followed by a single calibration fitting using the pre-fit model on a completely separate calibration training data set.
