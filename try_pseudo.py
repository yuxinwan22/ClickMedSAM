for i in range(3):
    pred, error_area = model(click)
    click = select_click(error_area)
loss = criterion(pred, tgt)

preds = []
for click in clicks:
    pred, error_area = model(click)
    preds.append(pred)
pred = voting(preds)
