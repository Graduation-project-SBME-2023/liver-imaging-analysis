import json

class Config:

    def __init__(self) -> None:
        with open("config/configs.json") as json_path:
            config_dict = json.load(json_path)
        self.device= config_dict["device"]
        self.train_data_path = config_dict['dataset']['training']
        self.test_data_path = config_dict['dataset']['testing']

        self.batch_size =   config_dict["training"]['batch_size']
        self.train_valid_split =   config_dict["training"]['train_valid_split']

        self.epochs = config_dict["training"]['epochs']
        self.loss_function = config_dict["training"]["loss_name"]
        self.loss_params= config_dict["training"]["loss_params"]
        self.optimizer = config_dict['training']['optimizer']
        self.optimizer_parameters = config_dict["training"]["optimizer_params"]

        self.resize = config_dict["transforms"]["transformation_size"]
        # self.apply_transform = config_dict["transforms"]["apply_transform"]
        self.img_key = config_dict["transforms"]["img_key"]
        self.label_key = config_dict["transforms"]["label_key"]
        self.tranform_name = config_dict["transforms"]["transform_name"]

        self.network_name = config_dict["network_name"]
        self.network_parameters  = config_dict["network_params"]


        self.model_checkpoint = config_dict["save"]["model_checkpoint"]
        self.potential_checkpoint = config_dict["save"]["potential_checkpoint"]
        


config = Config()
