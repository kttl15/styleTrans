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

    def downloadFiles(self, fileList: list, override: bool = False):
        """[summary]
        main method that iterates through a list of files to download

        Arguments:
            fileList {list} -- [List of files to download from firebase storage]

        Keyword Arguments:
            override {bool} -- [Whether or not to override existing files] (default: {False})
        """
        if len(fileList) > 0:
            print(f"Downloading {len(fileList)} Files")
            for file in fileList:
                self.downloadPathName = f"/home/a/Desktop/downloaded/{file}"
                if override:
                    self._downloadFile(file)
                elif not pathlib.Path(self.downloadPathName).exists():
                    # checks to see if the file exists
                    # if not then download
                    self._downloadFile(file)
        else:
            print("No Files")

    def _downloadFile(self, f: str):
        """[summary]
            internal method to download a file from firebase storage

            Arguments:
                f {str} -- [file dir]
            """
        if not pathlib.Path(self.downloadPathName).is_dir():
            filepath = "/".join(self.downloadPathName.split("/")[:-1])
        filepath = pathlib.Path(filepath)
        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)
        blob = self.bucket.get_blob(f)
        blob.download_to_filename(self.downloadPathName)

    def checkFileExist(self, fileList: list) -> bool:
        """[summary]
        checks to see if content/style pair exists

        Arguments:
            fileList {list} -- [List of content/style dir]

        Returns:
            bool
        """
        for f in fileList:
            if not self.bucket.get_blob(f):
                return False
        return True

    def uploadFolder(self, uid: str, processName: str):
        """[summary]
        upload output files to firebase storage

        Arguments:
            uid {str} -- [uid of current process]
            processName {str} -- [name of process]
        """
        print("Uploading Files")
        outputPath = os.listdir(
            f"/home/a/Desktop/downloaded/images/{uid}/{processName}/output"
        )
        lenOutput = len(outputPath)
        for output in outputPath:
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
