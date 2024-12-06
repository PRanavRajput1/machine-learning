# 1. create a new file named Dockerfile
# we know that 3.8.12-slim is our base image and we want to base our image on.
# so write the following code


# setting the base image for our image
FROM python:3.8.12-slim  

#installing virtual environment in our image
RUN pip install pipenv

#now since our project uses pipfile and pipfile lock. hence we want to copy these files to our container

#it create a directory if it does not exist there and then navigates to this directory
WORKDIR /app   

# these are the two files that we want to copy to the working path indicated by "./"
COPY ["Pipfile","Pipfile.lock","./"] 


#Inside docker we can now create a virtual environment
# but since docker itself is isolated hence we dont have to install/create virtual environment.
#but we need to install all the dependencies from the Pipfile to the container/system python

# we use a special key that is system it will install without creating a virtual environment and then deploy the libraries that are in pipfile
RUN pipenv --deploy  

#Copying the predict and model file to the container
COPY ["predict_flask.py","model_C=1.0.bin","./"] 


#now once you built this image and then run it and then serve the model using waitress then it works and run runs on 9696 port but this port of the container needs to be exposed so that our host machine can interact with it.

EXPOSE 9696


#WE SPECIFY  the entrypoint as terminal when we run the docker run command and when we skip that then it launches python since that is the default entrypoint but what we want to do when we run the docker image is that we want to run our web service using waitress hence we need to override the entrypoint.
# since every word should be separate here using commas.Hence we have written the command like this
ENTRYPOINT ["waitress", "--listen=0.0.0.0:9696", "predict_flask:app"]
# Now we can build the image and and run the image as following 
#docker built -t zoomcamp-test .  #-t referes to tech and we gave the tech name zoomcamp-test and the dot (.) refers that build the image based on the image in current directory

# and then run the image as following
#docker run --it --rm --entrypoint:bash zoomcamp-test
#but since we already defined entry point so we can skip that also we did exposed the 9696 port of the docker image but we need to map it to the host machine hence now the final command will be like this

#docker run --it --rm -p 9696:9696
