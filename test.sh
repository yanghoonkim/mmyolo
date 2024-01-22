python tools/test.py nia/rtmdet_l_nia.py work_dirs/rtmdet_l_nia/epoch_5.pth \
--cfg-options test_evaluator.classwise=True \
--json-prefix results \
# --show-dir show_results
# 추론 결과를 저장된 이미지로 확인하고 싶을 경우 활성화 위의 주석 활성화
화