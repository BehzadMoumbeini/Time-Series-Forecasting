
#set up the virtual enviroment: 
python3 -m venv venv source venv/bin/activate


#install requirements

pip install -r requirements.txt

#train the model and save it

jupyter notebook Untitled.ipynb

#run the app
python app.py
