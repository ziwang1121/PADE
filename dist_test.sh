
python test.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('5')" TEST.WEIGHT '/data2/zi.wang/code/PartialReID-final/logs_duke/lr0008_b32_Process1_Model12_loss1/transformer_best.pth' OUTPUT_DIR './logs_duke/test_AO_0.2'

python test.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('7')" TEST.WEIGHT '/data2/zi.wang/code/PartialReID-final/logs_market/lr0008_b32_Process1_Model12_loss1/transformer_best.pth' OUTPUT_DIR './logs_market/test_AO_0.2'


python test.py --config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DEVICE_ID "('5')" TEST.WEIGHT '../logs/occ_duke_vit_transreid_stride/transformer_120.pth'