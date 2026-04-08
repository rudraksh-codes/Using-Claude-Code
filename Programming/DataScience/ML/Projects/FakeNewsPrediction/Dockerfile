#python 
FROM python:3.9.6

#workdir 
WORKDIR /app 

#copy 
COPY . . 

#install dependencies 
RUN pip install -r requirements.txt 

#port 
EXPOSE 8501

#cmd 
CMD ["streamlit", "run", "app.py"]