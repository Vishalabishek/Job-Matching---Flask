from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Function to load and prepare data from CSV
def load_and_prepare_data(file_path, skill_columns):
    try:
        job_data = pd.read_csv(file_path)
        for column in skill_columns:
            if column in job_data.columns:
                job_data[column] = job_data[column].replace({'Yes': 1, 'No': 0})
                job_data[column] = pd.to_numeric(job_data[column], errors='coerce').fillna(0).astype(int)
        return job_data
    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        raise

# Function to recommend jobs based on user skills
def recommend_jobs(user_skills, job_data, skill_columns, top_n=5):
    try:
        user_skill_vector = np.zeros(len(skill_columns))
        for skill in user_skills:
            if skill in skill_columns:
                user_skill_vector[skill_columns.index(skill)] = 1

        job_skill_vectors = job_data[skill_columns].values
        similarity_scores = cosine_similarity([user_skill_vector], job_skill_vectors).flatten()

        job_data['Similarity'] = similarity_scores
        recommended_jobs = job_data.sort_values(by='Similarity', ascending=False).head(top_n)
        return recommended_jobs[['Job Title', 'Similarity']].to_dict(orient='records')
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route for job recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Ensure we received JSON data
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        user_skills = data.get('skills', [])
        if not user_skills:
            return jsonify({'error': 'No skills provided'}), 400
        
        # Path to your job dataset
        file_path = 'output-removed.csv'

        # Skill columns as per your dataset
        skill_columns = ['Python', 'Java', 'C++', 'SQL', 'HTML', 'CSS', 'JavaScript', 'React', 
                         'Git', 'Agile', 'Machine Learning', 'Operating Systems', 'Version Control', 
                         'Cloud Platforms', 'Containerization', 'Data Structures & Algorithms', 
                         'API Development', 'Microservices Architecture', 'Cybersecurity', 'Big Data', 
                         'CI/CD Pipelines']

        # Load and prepare data
        job_data = load_and_prepare_data(file_path, skill_columns)

        # Generate recommendations
        recommended_jobs = recommend_jobs(user_skills, job_data, skill_columns, top_n=20)

        # Return the recommendations as a JSON response
        return jsonify(recommended_jobs)

    except Exception as e:
        # Catch any errors and return a 500 Internal Server Error response
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
