import firebase_admin
from firebase_admin import credentials, firestore
import json
import datetime
from collections import OrderedDict
from time import perf_counter, time, process_time, sleep
import numpy as np
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor


class FirestoreUtils:
    def __init__(self, config: dict):

        assert isinstance(config, dict)
        assert "serviceAccount" in config.keys()

        cred = credentials.Certificate(config["serviceAccount"])
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.processDict = {}
        self.executor = ThreadPoolExecutor()

    def getProcessList(self, returnList: bool, outputFileName: str = None):
        """[summary]
        db stucture: collection(images) -> document(uid) -> collection(process) -> document(processName)
        
        main method that builds the output dict and saved as a json object
        
        checks to see if uid has any unprocessed data before retriving records to save
        time

        Keyword Arguments:
            outputFile {str} -- [name of json file] (default: {None})
        """
        col_ref = self.db.collection("images").stream()
        self.processList = []
        event_loop = asyncio.get_event_loop()
        if event_loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            event_loop = asyncio.get_event_loop()
        try:
            # get uids
            for d in col_ref:
                uid = self.db.collection("images").document(d.id)
                doc = uid.get()  # 0.2, 0.002, 0.2

                # check to see if the uid has unprocessed data
                if doc.to_dict()["hasUnprocessedFlag"]:
                    event_loop.run_until_complete(self.main(uid))
        finally:
            event_loop.close()

        if not returnList:
            with open(outputFileName if outputFileName else "test.json", "w") as f:
                json.dump(self.processList, f)
        else:
            return self.processList

    async def main(self, uid):
        """[summary]
        asynchronously retrives each record from firestore database

        Arguments:
            uid {[DocumentSnapshot]} -- [a firestore document snapshot]
        """
        processes = uid.collection("process").stream()
        loop = asyncio.get_event_loop()
        res = [
            loop.run_in_executor(self.executor, self.getData, [uid, process])
            for process in processes
        ]
        completed, pending = await asyncio.wait(res)

    def getData(self, uid_process: list):
        """[summary]
        builds a list of processes for each uid
        

        Arguments:
            uid_process {list} -- [a list of uid and process name]
        """
        uid = uid_process[0]
        process = uid_process[1]
        processDoc = uid.collection("process").document(process.id).get().to_dict()
        if processDoc["runOnUpload"] and not processDoc["isProcessed"]:
            processDoc["uploadDate"] = processDoc["uploadDate"].isoformat()
            self.processList.append(processDoc)

    def updateFields(self, uid: str, processName: str):
        print("Updating Fields")
        outputPath = os.listdir(
            f"/home/a/Desktop/downloaded/images/{uid}/{processName}/output"
        )
        lenOutput = len(outputPath)
        outputDict = {}
        for i, output in enumerate(outputPath):
            outputDict.update(
                {f"locOutput_{i+1}": f"images/{uid}/{processName}/{output}"}
            )
        self.db.collection("images").document(uid).collection("process").document(
            processName
        ).update({"isProcessed": True, "locOutputs": outputDict})


if __name__ == "__main__":
    config = {"serviceAccount": "/home/a/Desktop/gan/serviceAccount.json"}
    firestore = FirestoreUtils(config)
    outputFile = "processDict.json"

    start = np.array([perf_counter(), time(), process_time()])
    firestore.getProcessList(returnList=False)
    end = np.array([perf_counter(), time(), process_time()])
    diff_time = end - start

    a = 0
    for uid in firestore.processDict.keys():
        a += len(firestore.processDict[uid])

    print(
        f"""
        {{
            perf: {diff_time[0].round(2)}, 
            time: {diff_time[1].round(2)}, 
            proc: {diff_time[2].round(2)},
            user: {len(firestore.processDict.keys())}
            len:  {a}
        }}
        """
    )
