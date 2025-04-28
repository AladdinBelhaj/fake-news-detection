from django.shortcuts import render
import re
import os
import torch
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
from .models import TweetPrediction
from trainBERT import GNNFakeNews

def extract_tweet_id(url):
    patterns = [
        r'twitter\.com/\w+/status/(\d+)',
        r'x\.com/\w+/status/(\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def run_fake_news_detection(tweet_id, dataset_name='gossipcop', feature='bert', model_type='GCN'):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, 'data', 'UPFD')
        model_path = os.path.join(current_dir, 'gnn_fakenews.pt')
        dataset = UPFD(dataset_path, dataset_name, feature, 'test', transform=ToUndirected())
        loader = DataLoader(dataset, batch_size=1)
        model = GNNFakeNews(model_type, dataset.num_features, 128, dataset.num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            return str(pred.item())
        return None
    except Exception as e:
        print(f"Error running fake news detection: {e}")
        return None

def index(request):
    context = {
        'result': None,
        'tweet_id': None
    }
    if request.method == 'POST':
        twitter_url = request.POST.get('twitter_url', '')
        tweet_id = extract_tweet_id(twitter_url)
        if tweet_id:
            try:
                prediction_obj = TweetPrediction.objects.get(tweet_id=tweet_id)
                result = prediction_obj.prediction
            except TweetPrediction.DoesNotExist:
                result = run_fake_news_detection(tweet_id)
                if result in ['0', '1']:
                    TweetPrediction.objects.create(tweet_id=tweet_id, prediction=result)
            context['result'] = result
            context['tweet_id'] = tweet_id
    return render(request, 'detector/index.html', context)
