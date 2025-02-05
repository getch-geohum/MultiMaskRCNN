# model_configs = {
#
# }

train_configs = {"save_dir":"/MaskRCNN_output/model_final",   # directory to save file
                 "root":"/data",       # data root where training data is saved
                 "lr":0.001,
                 "epochs":50,
                 "lr_scheduler":None,
                 "train_batch":12,
                 "valid_batch":12,
                 "ignore":"Minawao_june_2016",
                 "name":"model_first",
                 "validate":True,
                 "val_type":"loss",
                 "sample_type":"systematic",
                 "train_ratio":0.9,
                 "petience":10,
                 "weight":None,
                 "row":256,
                 "col":256
                 }

predict_configs = {"save_dir": "/MaskRCNN_output/preds_final_check",
                   "data_dir": "/data/Minawao_june_2016",
                   "weight": "/MaskRCNN_output/model_final/model_model_first_best_weight.pt",
                   "model": "/MaskRCNN_output/model_model_first.pt",
                   "name": "minawao", # name to save prediction as shapefile
                   "mask_treshold": 0.5,
                   "score_treshold": 0.6,
                   "mask_iou_treshold": 0.8,
                   "bbox_iou_treshold": 0.3,
                   "image_type": "chips",
                   "row":256,
                   "col":256
                   }

finetune_configs = {"save_dir":"/MaskRCNN_output/model_tuned",
                    "root":"/data",
                    "lr":0.0001,
                    "epochs":50,
                    "train_batch":12,
                    "valid_batch":12,
                    "name":"refined_model",
                    "validate":True,
                    "val_ratio":0.1,
                    "val_type":"loss",
                    "sample_type":"systematic",
                    "train_ratio":0.2,
                    "reserve_test_sample":True,
                    "petience":10,
                    "weight":"/MaskRCNN_output/model/model_model_first_best_weight.pt",
                    "row":256,
                    "col":256
                    }
