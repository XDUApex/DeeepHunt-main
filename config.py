gaia = {
    # dataset
    "dataset": "gaia",
    # # 加载metric的tmp.json中间文件（复制使用）
    # "dataset_dir": "../DataSet/GAIA",
    # "save_dir": "./Data/GAIA",
    # 加载其他的tmp.json中间文件
    "dataset_dir": "DataSet/new_GAIA/service/webservice2",
    "save_dir": "DataSet/new_GAIA/service/webservice2",
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "drain_config": {
    "drain_save_path": "DataSet/new_GAIA/service/webservice2/drain.bin",
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # label info
    "failures": "login memory file access",
    "services": "webservice mobservice dbservice redisservice logservice",
    "instances": "webservice1 webservice2 logservice1 logservice2 mobservice1 mobservice2 redisservice1 redisservice2 dbservice1 dbservice2",
    "label_type": "failure_type",
    "num_class": 4,
    # cuda
    "use_cuda": True,
    "gpu": 0,
    # training
    "optim": "AdamW",
    "epochs": 50,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    # model
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 2,
    "dropout": 0.3,
}
aiops22 = {
    # dataset
    "dataset": "aiops22",
    "dataset_dir": "../datasets/aiops2022-pre",
    "save_dir": "data/aiops22",
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "drain_config": {
        "drain_save_path": "data/aiops22/drain.bin",
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # label info
    "failures": "cpu io memory network process",
    "services": "adservice cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice",
    # "instances": "adservice-0 adservice-1 adservice-2 adservice2-0 cartservice-0 cartservice-1 cartservice-2 cartservice2-0 checkoutservice-0 checkoutservice-1 checkoutservice-2 checkoutservice2-0 currencyservice-0 currencyservice-1 currencyservice-2 currencyservice2-0 emailservice-0 emailservice-1 emailservice-2 emailservice2-0 frontend-0 frontend-1 frontend-2 frontend2-0 paymentservice-0 paymentservice-1 paymentservice-2 paymentservice2-0 productcatalogservice-0 productcatalogservice-1 productcatalogservice-2 productcatalogservice2-0 recommendationservice-0 recommendationservice-1 recommendationservice-2 recommendationservice2-0 shippingservice-0 shippingservice-1 shippingservice-2 shippingservice2-0",
    "instances":   "adservice adservice-0 adservice-1 adservice-2 adservice2-0 cartservice cartservice-0 cartservice-1 cartservice-2 cartservice2-0 checkoutservice checkoutservice-0 checkoutservice-1 checkoutservice-2 checkoutservice2-0 currencyservice currencyservice-0 currencyservice-1 currencyservice-2 currencyservice2-0 emailservice emailservice-0 emailservice-1 emailservice-2 emailservice2-0 frontend frontend-0 frontend-1 frontend-2 frontend2-0 paymentservice paymentservice-0 paymentservice-1 paymentservice-2 paymentservice2-0 productcatalogservice productcatalogservice-0 productcatalogservice-1 productcatalogservice-2 productcatalogservice2-0 recommendationservice recommendationservice-0 recommendationservice-1 recommendationservice-2 recommendationservice2-0 redis-cart-0 redis-cart2-0 shippingservice shippingservice-0 shippingservice-1 shippingservice-2 shippingservice2-0",
    "label_type": "failure_type",
    "num_class": 5,
    # cuda
    "use_cuda": True,
    "gpu": 0,
    # training
    "optim": "SGD",
    "epochs": 120,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    # model
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 1,
    "dropout": 0.35,
}
train_ticket = {
    # dataset
    "dataset": "train_ticket",

    # # 加载metric的tmp.json中间文件（复制使用）
    # "dataset_dir": "../dataSet/GAIA",
    # "save_dir": "./data/GAIA",

    # 加载其他的tmp.json中间文件
    "dataset_dir": "../DataSet/train-ticket-original",
    "save_dir": "./data/D3",

    
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "drain_config": {
    "drain_save_path": "../dataSet/gaia_service/service/webservice2/drain.bin",
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # label info
    "failures": "login memory file access",
    "services": "webservice mobservice dbservice redisservice logservice",
    "instances": "ts-assurance-service ts-auth-service ts-basic-service ts-config-service ts-contacts-service ts-food-map-service ts-food-service ts-notification-service ts-order-other-service ts-order-service ts-preserve-other-service ts-preserve-service ts-price-service ts-route-service ts-seat-service ts-security-service ts-station-service ts-ticketinfo-service ts-train-service ts-travel-service ts-travel2-service ts-ui-dashboard ts-user-service ts-verification-code-service",
    "label_type": "failure_type",
    "num_class": 4,
    # cuda
    "use_cuda": True,
    "gpu": 0,
    # training
    "optim": "AdamW",
    "epochs": 50,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    # model
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 2,
    "dropout": 0.3,
}

CONFIG_DICT = {"gaia": gaia, "aiops22": aiops22, "train_ticket": train_ticket}
