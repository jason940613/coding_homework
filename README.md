機器學習期末作業流程說明:
1.使用train.ipynb載入訓練資料集（影像與對應之 YOLO 標註檔），訓練YOLO12n物件偵測模型。
2.以訓練完成後所產生之最佳權重（best.pt），對測試資料集進行推論，並輸出YOLO格式之預測標註結果。
3.使用測試程式eval_yolo_custom.py，比對模型推論結果與真實標註資料。
4.根據IoU門檻值進行配對，產生True Positive（TP）、False Positive（FP）與 False Negative(FN)等分析結果。