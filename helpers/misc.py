import os
import time


def get_model_name():
    model_name = ""

    while True:
        model_name = input("How do you want to call your model ?\n")

        if not os.path.exists(f"models/{model_name}.hdf5"):
            break

        print(f"Model with name {model_name} already exists !")

    return model_name


def get_timestamp(elapsed_time):
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
