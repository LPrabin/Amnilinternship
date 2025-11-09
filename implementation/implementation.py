from fastapi import FastAPI, File, Form, UploadFile

#lets think sequentially from to import images to process images to return images






app = FastAPI()


@app.get("input/")
async def read_input(file : UploadFile = File(...)):
    #here we will write code to accept images from user ,may be single image or folder of images
     
