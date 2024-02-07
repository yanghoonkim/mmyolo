python tools/test.py nia/rtmdet_l_nia.py work_dirs/finetuned/epoch_5.pth \
#--cfg-options test_evaluator.classwise=True \
# evaluation시 classwise AP를 확인하고 싶은 경우 활성화
#--json-prefix results \
# json-prefix: the prefix of the output json file without perform evaluation 
#--show-dir show_results
# show-dir: 추론 결과를 저장된 이미지로 확인하고 싶을 경우 활성화
