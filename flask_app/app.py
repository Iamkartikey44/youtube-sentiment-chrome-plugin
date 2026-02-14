import os
import io
import re
import joblib
import mlflow
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from flask import Flask,request,jsonify,send_file
from flask_cors import CORS
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


matplotlib.use('Agg')

app = Flask(__name__)
CORS(app) #Enables CORS for all routes

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n',' ',comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]','',comment)
        stop_words = set(stopwords.words('english')) - {'not','but','however','no','yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        raise

def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("https://dagshub.com/Iamkartikey44/youtube-sentiment-chrome-plugin.mlflow")  # Replace with your MLflow tracking URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Load assets
model, vectorizer = load_model_and_vectorizer("lgbm_model", "1", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to our flask api"   

@app.route("/predict_with_timestamps", methods=['POST'])
def predict_with_timestamps():
    try:
        data = request.get_json(force=True)

        if not data or "comments" not in data:
            return jsonify({'error': "No comments provided"}), 400

        comments_data = data.get('comments')

        if not isinstance(comments_data, list) or len(comments_data) == 0:
            return jsonify({'error': "Comments must be a non-empty list"}), 400

        # Extract text and timestamps safely
        comments = []
        timestamps = []

        for item in comments_data:
            if 'text' not in item or 'timestamp' not in item:
                return jsonify({'error': "Each item must contain 'text' and 'timestamp'"}), 400
            comments.append(item['text'])
            timestamps.append(item['timestamp'])

        # Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # 🔥 Convert to DataFrame (CRITICAL FIX)
        transformed_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict
        predictions = model.predict(transformed_df)
        predictions = [str(pred) for pred in predictions]

        # Build response
        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "timestamp": timestamp
            }
            for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # ---------------------------
        # Step 1: Preprocess
        # ---------------------------
        preprocessed_comments = [
            preprocess_comment(comment) for comment in comments
        ]

        # ---------------------------
        # Step 2: Vectorize
        # ---------------------------
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # ---------------------------
        # Step 3: Convert to DataFrame (CRITICAL FOR MLFLOW)
        # ---------------------------
        transformed_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # ---------------------------
        # Step 4: Predict (use DataFrame, NOT sparse matrix)
        # ---------------------------
        predictions = model.predict(transformed_df)

        # Convert to string (same output format as before)
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # ---------------------------
    # Step 5: Return same output format
    # ---------------------------
    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]

    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({'error': 'No sentiment counts provided'}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]

        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        plt.close()

        img_io.seek(0)  # 🔥 CRITICAL

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud',methods=['POST'])
def generate_wordcloud():

    try:
        data =request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400
        
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(width=800,height=800,background_color='black',colormap='Blues',stopwords=set(stopwords.words('english')),collocations=False).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io,format='PNG')
        img_io.seek(0)

        return send_file(img_io,mimetype='image/png')
    
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500
    

@app.route('/generate_trend_graph',methods=['POST'])
def generate_trend_graph():

    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
           return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp',inplace=True)

        sentiment_labels = {-1:'Negative', 0:'Neutral', 1:'Positive'} 
        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)  # Calculate total counts per month
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100 # Calculate percentages

        for sentiment_value in [-1,0,1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1,0,1]] # Sort columns by sentiment value

        plt.figure(figsize=(12,6))
        colors= {-1:'red',0:'gray',1:'green'}

        for sentiment_value in [-1,0,1]:
            plt.plot(monthly_percentages.index,monthly_percentages[sentiment_value],marker='o',linestyle='-',label=sentiment_labels[sentiment_value],
                     color=colors[sentiment_value])
        
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io,format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io,mimetype='image/png')
    
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5000, debug=False)




