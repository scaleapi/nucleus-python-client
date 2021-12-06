def test_list_eval_functions(CLIENT):
    eval_functions = CLIENT.modelci.eval_functions
    print(eval_functions.bbox_iou > 0.5)
