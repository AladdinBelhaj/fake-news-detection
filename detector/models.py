from django.db import models

# Create your models here.

class TweetPrediction(models.Model):
    tweet_id = models.CharField(max_length=50, unique=True)
    prediction = models.CharField(max_length=1)

    def __str__(self):
        return f"Tweet ID: {self.tweet_id} - Prediction: {self.prediction}"
