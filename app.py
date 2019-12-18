from flask import Flask,render_template,url_for,request
#from sklearn.externals import joblib
import joblib
import pickle
from nltk.corpus import stopwords 
stopwords = set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    tfidf_vect_pkl = open('tfidf_vectorizer.pkl','rb')
    tfidf_vect = joblib.load(tfidf_vect_pkl)
    clf_pkl = open('svm_model.pkl','rb')
    clf = joblib.load(clf_pkl)

    if request.method == 'POST':
        veggies = request.form['veggies']
        grains = request.form['grains']
        spices = request.form['spices']
        oils = request.form['oils']
        meats = request.form['meats']
        dairy = request.form['dairy']
        extras = request.form['extras']
        store_items=[veggies,grains,spices,oils,meats,dairy,extras]
        if(len(store_items[0]) > 0 or len(store_items[1]) > 0 or len(store_items[2]) > 0 or len(store_items[3]) > 0 or len(store_items[4]) > 0 or len(store_items[5]) > 0 or len(store_items[6]) > 0):
            store_item_str=''
            for item in store_items:
                if(len(item)>0):
                    store_item_str=item+' '+store_item_str
            store_item_str=store_item_str.replace("'","")
            store_item_str=store_item_str.replace(",","")
            store_item_str=store_item_str.replace("-"," ")
            store_item_str=store_item_str.replace("_","")
            sentance=' '.join(e.lower() for e in store_item_str.split() if e.lower() not in stopwords)
            print(sentance)
            store_items_list=[sentance]
            test_data_vect=tfidf_vect.transform(store_items_list)
            clf_prediction = clf.predict(test_data_vect)
        else:
            clf_prediction=['Enter atleast more than one ingredients']
    return render_template('results.html',prediction = clf_prediction[0])
    
if __name__ == '__main__':
	app.run(debug=True)
