from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
import json
import pathlib
from firebase_admin import credentials


class FirebaseStorage:
    def __init__(self, config: dict):
        assert isinstance(config, dict)
        assert "serviceAccount" in config.keys()
        serviceAccountJSON = config["serviceAccount"]

        with open(serviceAccountJSON, "r") as f:
            temp = json.load(f)
            storage_bucket = temp["project_id"] + ".appspot.com"
        scopes = [
            "https://www.googleapis.com/auth/firebase.database",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            serviceAccountJSON, scopes
        )
        client = storage.Client(credentials=credentials, project=storage_bucket)
        self.bucket = client.get_bucket(storage_bucket)

    def checkFilePathExist(self, path: str):
        if not pathlib.Path(path).is_dir():
            path = "/".join(path.split("/")[:-1])
        path = pathlib.Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def downloadFile(self, storageFilename: str, downloadPathName: str):
        self.checkFilePathExist(downloadPathName)
        blob = self.bucket.get_blob(storageFilename)
        blob.download_to_filename(downloadPathName)


config = {
    "storageBucket": "flutter-bloc-1-2eb17.appspot.com",
    "serviceAccount": "/home/a/Desktop/gan/flutter-bloc-1-2eb17-firebase-adminsdk-mhniq-a589c3de7a.json",
}

firebase = FirebaseStorage(config)
firebase.downloadFile(
    storageFilename="images/UPjcLOcMuKfLHeeEeJhF5IVHvCP2/cncx/content.jpg",
    downloadPathName="downloaded/images/UPjcLOcMuKfLHeeEeJhF5IVHvCP2/cncx/content.jpg",
)
