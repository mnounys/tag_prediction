from flask import Flask, request, url_for, render_template, redirect, flash
import sys
import os
sourceDir = os.path.join(os.path.dirname(__file__), 'source')
sys.path.append(sourceDir)
from TagPrediction import TagPrediction
import json

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'
pred = TagPrediction()

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        lTitle = request.form['title']
        lBody = request.form['content']

        if not lTitle:
            flash('Title is required!')
        elif not lBody:
            flash('Body is required!')
        else:
            return redirect(url_for('getPrediction',Title = lTitle, Body = lBody ))

    return render_template('submit.html')

@app.route('/predict/', methods=['GET'])
def getPrediction():
    
    if "Title" not in request.args :
        return 'Error : Title parameter is mandatory'
    
    if "Body" not in request.args :
        return 'Error : Body parameter is mandatory'
    
    lTitle = str(request.args['Title'])
    lBody = str(request.args['Body'])
     
    lRes = [{ 'title' : lTitle,
             'body' : lBody,
             'tags' : pred.getPrediction(lTitle,lBody)
            }
             ]
    
    return render_template('result.html', messages = lRes, index_url = url_for('index'))

@app.route('/predictResOnly/', methods=['GET'])
def getPredictionOnly():
    
    if "Title" not in request.args :
        return 'Error : Title parameter is mandatory'
    
    if "Body" not in request.args :
        return 'Error : Body parameter is mandatory'
    
    lTitle = str(request.args['Title'])
    lBody = str(request.args['Body'])
     
    lRes = pred.getPrediction(lTitle,lBody)
    
    return json.dumps(lRes)


@app.route('/tags/', methods=['GET'])
def getMetadata():
    listtags = pred.getTagList()
    return json.dumps(listtags.tolist())
    
    
if __name__ == "__main__":
    app.run()  # run our Flask app