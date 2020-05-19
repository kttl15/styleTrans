from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
import json
import pathlib
import os
from firebase_admin import credentials


class FirebaseStorageUtils:
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

    def downloadFiles(self, storageFilename: list, override: bool = False):
        for s in storageFilename:
            self.downloadPathName = f"/home/a/Desktop/downloaded/{s}"
            if override:
                self._downloadFile(s)
            elif not pathlib.Path(self.downloadPathName).exists():
                self._downloadFile(s)

    def _downloadFile(self, s: str):
        if not pathlib.Path(self.downloadPathName).is_dir():
            filepath = "/".join(self.downloadPathName.split("/")[:-1])
        filepath = pathlib.Path(filepath)
        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)
        blob = self.bucket.get_blob(s)
        blob.download_to_filename(self.downloadPathName)

    def uploadFolder(self, uid: str, processName: str):
        outputPath = os.listdir(
            f"/home/a/Desktop/downloaded/images/{uid}/{processName}/output"
        )
        lenOutput = len(outputPath)
        for output in outputPath:
            print(
                f"images/{uid}/{processName}/{output}",
                f"/home/a/Desktop/downloaded/images/{uid}/{processName}/output/{output}",
            )
            blob = self.bucket.blob(f"images/{uid}/{processName}/{output}")
            blob.upload_from_filename(
                f"/home/a/Desktop/downloaded/images/{uid}/{processName}/output/{output}"
            )

    def deleteFile(self, storageFileName: str):
        self.bucket.delete_blob(storageFileName)


if __name__ == "__main__":
    config = {
        "storageBucket": "flutter-bloc-1-2eb17.appspot.com",
        "serviceAccount": "serviceAccount.json",
    }

    firebase = FirebaseStorageUtils(config)
    firebase.downloadFiles(
        storageFilename=[
            "images/UPjcLOcMuKfLHeeEeJhF5IVHvCP2/cjcjc/content.jpg",
            "images/UPjcLOcMuKfLHeeEeJhF5IVHvCP2/cjcjc/style.jpg",
        ],
    )
