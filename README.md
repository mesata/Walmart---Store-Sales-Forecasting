# Walmart---Store-Sales-Forecasting
ეს არის მანქანური სწავლების (Machine Learning) პროექტი, რომლის მიზანია **Walmart**-ის მაღაზიების და დეპარტამენტების ყოველკვირეული გაყიდვების პროგნოზირება ისტორიული მონაცემების, სეზონურობისა და სხვადასხვა ეკონომიკური ფაქტორების (CPI, Unemployment და ა.შ.) საფუძველზე.

--------------------------------------------------------------------------------------------
# რეპოზიტორიის სტრუქტურა
---

## EDA 

# Experiment Tracking - Linear models

## XGBoost
ამ მოდელის გაწვრთნისას თავიდან დავიწყე base feature engineering-ით (WalmartTransformer კლასი), რომელიც მოიცავდა მარტივი feature-ების შექმნით: Year, Month, Day. ეს გავაკეთე, რადგან ერთი დროის feature დამეშალა და მოდელისთვის მეტი საშუალება მიმეცა დროის აღქმისთვის. ამ feature engineering-ზე დაყრდნობით გავუშვი შემდეგი run-ები:

| მოდელის ტიპი | პარამეტრები | Train MAE | Validation MAE |
| :--- | :--- | :--- | :--- | 
| **XGBoost (Baseline)** | `n_estimators=100`, `max_depth=6`, `lr=0.1` | $3,898.24 | $3,935.79 |
| **XGBoost (Tuned)** | `n_estimators=300`, `max_depth=7`, `lr=0.03`, `subsample=0.8` | $3,598.26 | **$3,711.25** | 
| **XGBoost (Advanced Tuned)** | `n_estimators=400`, `max_depth=8`, `lr=0.05`, `gamma=0.2`, `min_child_weight=3` | $2,327.12 | **$2,818.29** | 
---

პარამეტრების ოპტიმიზაციის შედეგეად შედეგი უმჯობესდებოდა, Validation MAE $3,935.79 დავიდა $2,818.29-ზე. 
მაგრამ შედეგი მაინც არ მეჩვენა საკმარისი და ვცადე შემეცვალა Feature Engineering მიდგომა. გავაკეთე ახალი WalmartDataTransformer_updatedFeatureEngineering კლასი, დავამატე ახალი isDecember და isNovember feature-ები, რადგან როგორც დატაზე დაკვირვებისას ვნახეთ ამ პერიოდში გაყიდვები ყველაზე მეტად იზრდებოდა და მინდოდა მოდელს ამ კანონზომიერებისთვისაც მიექცია ყურადღება. 

| მოდელის ტიპი | პარამეტრები | Train MAE | Validation MAE |
| :--- | :--- | :--- | :--- | 
| **XGBoost (Advanced + FE)**| **Advanced + Holiday & Monthly Features** | $2,223.33 | **$2,697.30** |
