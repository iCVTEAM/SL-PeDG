import logging


def split_dg_dataset(dataset_name):
    logger = logging.getLogger(__name__)

    if 'VIPER' in dataset_name:
        dg_dataset_name = 'DG_VIPER'
        sub_num = 1 if 'only' in dataset_name else 4

        try:
            num_test = int(dataset_name.split('_')[-1])
        except TypeError:
            num_test = 10

        sub_type = ['c', 'd', 'a', 'b']
        total_sub_names = [["split_" + str(i + 1) + x for i in range(num_test)] for j, x in enumerate(sub_type)]

        sub_names = []
        for i in range(sub_num):
            sub_names.extend(total_sub_names[i])

        logger.info("Evaluating in {} VIPER subsets.".format(num_test))

    elif 'PRID' in dataset_name:
        dg_dataset_name = 'DG_PRID'
        sub_names = [x for x in range(10)]
        logger.info("Evaluating in 10 PRID subsets.")
    elif 'GRID' in dataset_name:
        dg_dataset_name = 'DG_GRID'
        sub_names = [x for x in range(10)]
        logger.info("Evaluating in 10 GRID subsets.")
    elif 'ILIDS' in dataset_name:
        dg_dataset_name = 'DG_ILIDS'
        sub_names = [x for x in range(10)]
        logger.info("Evaluating in 10 iLIDS subsets.")

    else:
        dg_dataset_name = dataset_name
        sub_names = ['']

    return dg_dataset_name, sub_names
