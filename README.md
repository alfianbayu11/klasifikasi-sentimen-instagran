**Description**

**Instagram Comment Sentiment Website** is an interactive web application built with Streamlit to predict the sentiment of comments from Instagram. The application uses a pre-trained Support Vector Machine (SVM) model to classify comments as positive, negative, or neutral.

**Key Features:**

* **Comment Sentiment Prediction:** Enter an individual comment or upload a CSV file containing a list of comments to get sentiment predictions for each comment.
* **Data Visualization:** Visualize the distribution of comment sentiment using interactive pie charts and wordclouds.
* **Data Export:** Export comment data and sentiment predictions to a CSV file for further analysis.

**How to Use:**

1. Run the Streamlit application with the command `streamlit run app.py`.
2. Select the "Sentiment Prediction" menu to perform individual predictions or "Data Visualization" to view the collected data.
3. For individual predictions, enter a comment in the provided text field and click the "Submit" button.
4. For batch predictions from a CSV file, upload the file and select the columns that contain the comments and sentiment (if applicable).
5. On the data visualization page, you can view a pie chart of sentiment distribution and a wordcloud that shows the keywords that frequently appear in the comments.

**Requirements:**

* Python 3.7+
* Streamlit
* scikit-learn
* Pillow
* WordCloud
* PyMySQL (for SQLite database connection)

**Advanced Development:**

* Integration with the Instagram API for direct comment retrieval.
* Development of more sophisticated sentiment models using deep learning.
* Sentiment analysis based on specific aspects of the comments (e.g., product, service, price).

**Repository:**

The application is available in the following Github repository: [https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/blob/master/dataset_komentar_instagram_cyberbullying.csv](https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/blob/master/dataset_komentar_instagram_cyberbullying.csv): [https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/blob/master/dataset_komentar_instagram_cyberbullying.csv](https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/blob/master/dataset_komentar_instagram_cyberbullying.csv)

**Contributions:**

We welcome community contributions to the development of this application. Please fork the repository and submit pull requests with features or fixes that you create.

**We hope this application is useful!**




