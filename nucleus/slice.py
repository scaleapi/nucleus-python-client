from .constants import DATASET_ITEM_ID_KEY
import requests
import grequests


class Slice:
    """
    Slice respesents a subset of your Dataset.
    """

    def __init__(self, slice_id: str, client):
        self.slice_id = slice_id
        self._client = client

    def info(self, id_type: str = DATASET_ITEM_ID_KEY) -> dict:
        """
        This endpoint provides information about specified slice.

        :param
        slice_id: id of the slice
        id_type: the type of IDs you want in response (either "reference_id" or "dataset_item_id")
        to identify the DatasetItems

        :return:
        {
            "name": str,
            "dataset_id": str,
            "dataset_item_ids": List[str],
        }
        """
        return self._client.slice_info(self.slice_id, id_type)

    def download(self, id_type):
        """
        This function downloads all images pertaining to a given slice in the current directory.

        :param
        id_type: the type of IDs you want in response (either "reference_id" or "dataset_item_id")
        """
        assert(id_type == 'reference_id' or id_type == 'dataset_item_id')

        slice_info = self._client.slice_info(self.slice_id, id_type)
        dataset_id = slice_info.get("dataset_id", "")
        dataset_item_ids = slice_info.get(f'{id_type}s', "")


        print("Downloading Slice..")

        async_requests = []
        for dataset_item in dataset_item_ids:
            request = self._client.dataitem_ref_id(
                dataset_id, dataset_item, True, session=requests.session()) if id_type == 'reference_id' else self._client.dataitem_loc(dataset_id, dataset_item, True, sess)

            async_requests.append(request)

        def exception_handler(request, exception):
            print("An exception occured: ", request, exception)

        responses = grequests.map(async_requests, exception_handler=exception_handler)

        print("Saving images to current dir..")

        for index, response in enumerate(responses):
            image_type = response.headers.get("Content-Type").split('/')[-1]
            file_name = f'{dataset_item_ids[index]}.{image_type}'
            image_file = open(f'{file_name}', 'w+b')
            image_file.write(response.content)
            image_file.close()

        print("Done!")
