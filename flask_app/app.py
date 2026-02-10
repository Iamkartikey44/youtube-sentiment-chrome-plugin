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

def load_model_and_vectorizer_local(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# Load assets
model, vectorizer = load_model_and_vectorizer_local(
    "./lgbm_model.pkl",
    "./tfidf_vectorizer.pkl"
)

@app.route('/')
def home():
    return "Welcome to our flask api"   

@app.route("/predict_with_timestamps",methods=['POST'])
def predict_with_timestamps():
    data  = request.json
    comments_data =data.get('comments')
     
    if not comments_data:
        return jsonify({'error':"No comments provided"}),400
    
    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions =model.predict(transformed_comments).tolist()
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({'error':f"Prediction failed: {str(e)}"}),500   
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)


@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comment')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    
    try:

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments).tolist()
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
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

    app.run(host='0.0.0.0',port=5000,debug=True)




