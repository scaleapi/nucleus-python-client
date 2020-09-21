class Dataset:
    """
    Nucleus Dataset. You can append images with metadata to your dataset,
    annotate it with ground truth and upload model predictions to evaluate and
    compare model performance on you data.
    """

    def __init__(self, dataset_id: str, client):
        self.dataset_id = dataset_id
        self._client = client

    @property
    def info(self) -> dict:
        """
        Returns information about existing dataset
        :return: dictionary of the form
            {
                'name': str,
                'length': int,
                'model_run_ids': List[str],
                'slice_ids': List[str]
            }
        """
        return self._client.dataset_info(self.dataset_id)

    def create_model_run(self, payload: dict):
        """
        Creates model run for the dataset based on the given parameters:

        'reference_id' -- The user-specified reference identifier to associate with the model.
                        The 'model_id' field should be empty if this field is populated.

        'model_id' -- The internally-controlled identifier of the model.
                    The 'reference_id' field should be empty if this field is populated.

        'name' -- An optional name for the model run.

        'metadata' -- An arbitrary metadata blob for the current run.

        :param payload:
        {
            "reference_id": str,
            "model_id": str,
            "name": Optional[str],
            "metadata": Optional[Dict[str, Any]],
        }
        :return:
        {
          "model_id": str,
          "model_run_id": str,
        }
        """
        return self._client.create_model_run(self.dataset_id, payload)

    def annotate(self, payload: dict) -> dict:
        """
        Uploads ground truth annotations for a given dataset.
        :param payload: {"annotations" : List[Box2DAnnotation]}
        :return:
        {
            "dataset_id: str,
            "new_items": int,
            "updated_items": int,
            "ignored_items": int,
        }
        """
        return self._client.annotate_dataset(self.dataset_id, payload)

    def append(self, payload: dict, local=False) -> dict:
        """
        Appends images with metadata (dataset items) to the dataset. Overwrites images on collision if forced.

        :param payload: {"items": List[DatasetItem], "force": bool}
        :return:
        {
            'dataset_id': str,
            'new_items': int,
            'updated_items': int,
            'ignored_items': int,
        }
        """
        return self._client.populate_dataset(self.dataset_id, payload, local=local)

    def iloc(self, i: int) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Number.
        :param i: absolute number of dataset item for the given dataset.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Box2DAnnotation],
        }
        """
        return self._client.dataitem_iloc(self.dataset_id, i)

    def refloc(self, reference_id: str) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Reference Id.
        :param reference_id: reference_id of dataset item.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Box2DAnnotation],
        }
        """
        return self._client.dataitem_ref_id(self.dataset_id, reference_id)

    def loc(self, dataset_item_id: str) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Id.
        :param dataset_item_id: internally controlled id for the dataset item.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Box2DAnnotation],
        }
        """
        return self._client.dataitem_loc(self.dataset_id, dataset_item_id)
