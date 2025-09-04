# lazada_astaxanthin_machine_learning
continue the Web Scraping with python visualization and machine learning


## Cell 1 — Imports (and auto-install if needed)

### (what & why):

Brings in Python libraries for data handling ( ```pandas```, ```numpy``` ), charts (```matplotlib```), Drive/Sheets access, and machine learning (```scikit-learn```).

If Colab doesn’t have certain Google APIs, the cell installs them automatically so the rest of the notebook runs smoothly.

### (คืออะไร & ทำไม):

นำเข้าไลบรารีสำหรับจัดการข้อมูล ( ```pandas```, ```numpy```), วาดกราฟ (```matplotlib```), ต่อ Google Drive/Sheets และทำ Machine Learning (```scikit-learn```)

ถ้า Colab ยังไม่มีบางแพ็กเกจ จะติดตั้งให้อัตโนมัติ เพื่อให้โน้ตบุ๊กรันต่อได้ไม่มีสะดุด

## Cell 2 — Mount Drive & load data (supports .gsheet shortcut)

Connects your Colab to Google Drive.

If your file is a Google Sheets shortcut (```.gsheet```), it resolves the shortcut to the real spreadsheet and loads the first worksheet into a DataFrame. If it’s a CSV, it reads it directly.

Prints shape and columns so you can confirm the data loaded correctly.



เชื่อม Colab เข้ากับ Google Drive

ถ้าไฟล์เป็น Shortcut ของ Google Sheets (```.gsheet```) จะค้นหาไอดีของชีตตัวจริงแล้วโหลดแผ่นงานแรกเข้ามาเป็น DataFrame ถ้าเป็น CSV ก็อ่านตรงๆ

แสดงขนาดตารางและชื่อคอลัมน์เพื่อยืนยันว่าโหลดถูกต้อง

## Cell 3 — Cleaning & parsing helpers



- Defines small functions to turn messy text into numbers:

  - ```parse_price``` understands “฿1,990”, “149–199” (takes the average).

  - ```parse_number_with_suffix``` handles counts like “1.2k”, “3m”, or “1,234 sold”.

Keeps your numeric features clean and comparable.



- ฟังก์ชันแปลงข้อความเป็นตัวเลข:

  - ```parse_price``` เข้าใจราคาแบบมีสัญลักษณ์/ช่วง เช่น “฿1,990”, “149–199” (เอาค่าเฉลี่ย)

  - ```parse_number_with_suffix``` เข้าใจจำนวนแบบ “1.2k”, “3m”, หรือ “1,234 sold”

เพื่อให้ฟีเจอร์ตัวเลขสะอาดและเทียบกันได้

## Cell 4 — Build cleaned dataset (name, price, sold_est)


Auto-detects which columns in your file match ```name```, ```price```, and ```sold_est```.

Converts ```price``` and ```sold_est``` to numeric, drops invalid rows, and keeps a clean table with exactly ```name```, ```price```, ```sold_est```.

This is your analysis-ready product table.


ตรวจหาคอลัมน์ที่ตรงกับ ```name```, ```price```, ```sold_est```

แปลง ```price``` and ```sold_est``` เป็นตัวเลข ตัดแถวที่ไม่ถูกต้อง เหลือตารางสะอาดเฉพาะ ```name```, ```price```, ```sold_est```

ได้ตาราง พร้อมวิเคราะห์ สำหรับสินค้า

## Cell 5 — Feature engineering (log transforms + light text parsing)


Creates better-behaved numeric features:

  - ```price_log = log1p(price)``` and ```sold_log = log1p(sold_est)``` compress big extremes so clustering isn’t dominated by a few huge values.

  - Very light text parsing from ```name``` (e.g., extract “mg” dosage or pack size if present) to add subtle product differences.

Scales features with ```RobustScaler``` to reduce outlier effect.


สร้างฟีเจอร์เชิงตัวเลขที่ “ดูดีขึ้น”:

  - ```price_log = log1p(price)``` และ ```sold_log = log1p(sold_est)``` เพื่อบีบค่าที่กระโดด ไม่ให้ข้อมูลบางจุดครอบงำการจัดกลุ่ม

  - แยกข้อมูลเล็กน้อยจาก ```name``` (เช่น ปริมาณ mg หรือจำนวนเม็ด/แพ็ก ถ้าพบ) เพื่อเพิ่มรายละเอียดผลิตภัณฑ์

ใช้ ```RobustScaler``` เพื่อลดผลกระทบจาก outlier


## Cell 6 — Choose number of clusters (k) with silhouette score

### (simple idea):

Tries several k values (e.g., 3–8) and for each one measures how clearly the groups separate using silhouette score (0 to 1; higher is cleaner separation).

Picks the k that gives the best score and prints it: ```Chosen k = ....```

### (อธิบายง่ายๆ):

ทดลองจำนวนกลุ่ม k หลายค่า (เช่น 3–8) แล้ววัดความ “แยกชัด” ของกลุ่มด้วย silhouette score (0 ถึง 1; ยิ่งสูงยิ่งแยกชัด)

เลือก k ที่คะแนนดีที่สุด และพิมพ์ผล: ```Chosen k = ...```

✅ How many segments? Look at the output line from this cell:

If it prints, say, ```Chosen k=4```, then you have 4 segments.

If several ks are close, choose the one that makes business sense (easier to explain/act on, e.g., 3–5 segments).

## Cell 7 — Fit best K-Means and quick profile


Uses the chosen k to assign each product to a cluster (```cluster_km```).

Builds a quick profile table per cluster (count of products, median price, median sold, 75th percentiles).

This table helps you name the segments (e.g., “Premium Star”, “Mass-Market Hero”, etc.).


ใช้ค่า k ที่เลือก เพื่อจัดสินค้าลงกลุ่ม (```cluster_km```)

ทำตาราง สรุปโปรไฟล์ รายกลุ่ม (จำนวนสินค้า ราคากลาง ยอดขายกลาง และเปอร์เซ็นไทล์ที่ 75)

ตารางนี้ช่วยให้คุณ ตั้งชื่อ segment ได้ (เช่น “Premium Star”, “Mass-Market Hero” ฯลฯ)

## Cell 8 — Scatter plot: price vs sold_est (colored by cluster)


Plots the actual business axes you know: Price vs Estimated Sold, and colors each point by cluster.

Quick visual to see where each segment lives (e.g., high-price/low-sold vs low-price/high-sold).


กราฟราคากับยอดขายที่คุณคุ้นเคย: ราคา vs ยอดขายประมาณการ โดยลงสีตามกลุ่ม

ช่วยเห็นภาพรวมว่าแต่ละกลุ่มอยู่โซนไหน (เช่น ราคาสูง/ขายน้อย vs ราคาต่ำ/ขายเยอะ)

## Cell 9 — PCA 2D (just for visualization)


Reduces the feature space down to 2D only for plotting (PCA).

Not used by K-Means; it’s just a map to help you see that clusters are reasonably separated.


ย่อตัวแปรหลายตัวให้เหลือ 2 มิติ เพื่อวาดรูปเท่านั้น (PCA)

ไม่ได้ใช้ในการจัดกลุ่ม เป็นแผนที่ให้เห็นว่ากลุ่มแยกกันพอควร

## Cell 10 — Save segmented table to Drive


Exports a CSV with the key fields and assigned cluster label, so you can use it in Looker Studio/Power BI: ```name, price, sold_est, cluster_km```.


ส่งออก CSV ที่มีคีย์ฟิลด์และป้ายกลุ่ม เพื่อนำไปใช้ต่อใน Looker Studio/Power BI: ```name, price, sold_est, cluster_km```




## Why K-Means? (simple, business-friendly)


You don’t have “true labels” for product types—so we use unsupervised learning.

K-Means is fast, simple, and great when you want to group similar items by numbers like price and sales.

It finds groups by pulling items that are close together (after we log-transform & scale so no single column dominates).


เราไม่มี “ป้ายคำตอบจริง” ว่าสินค้าอยู่กลุ่มไหน จึงใช้วิธี ไม่กำกับดูแล (unsupervised)

K-Means เร็ว เรียบง่าย เหมาะกับการ จัดกลุ่มสินค้าที่คล้ายกัน ด้วยตัวเลข เช่น ราคาและยอดขาย

หลักการคือดึงสินค้าที่ ใกล้กัน มาอยู่กลุ่มเดียว (หลังทำ log/scale เพื่อลดอิทธิพลค่าสุดโต่ง)

## How many segments? (reading Cell 6)


The notebook prints ```Chosen k=...``` based on the highest silhouette score.

Typical business-friendly ranges: 3–6 segments. If scores are similar, pick what’s easiest to explain and act on (e.g., 4).


โน้ตบุ๊กจะแสดง ```Chosen k=...``` จากคะแนน silhouette ที่ดีที่สุด

ช่วงที่เข้าใจง่ายทางธุรกิจมักอยู่ 3–6 กลุ่ม ถ้าคะแนนสูสีกัน เลือกจำนวนที่อธิบาย/นำไปใช้ได้ง่าย (เช่น 4)

## How are the segments different? (reading Cell 7 table)

### (example naming):
Use the cluster profile table (median price/sold, 75th percentiles) to name segments:

Premium Star — high price, high sold (rare but powerful)

Premium Niche — high price, modest sold

Mass-Market Hero — low-to-mid price, high sold

Long Tail / Mid — low-to-mid price, low-to-mid sold

### (ตัวอย่างตั้งชื่อ):
ตั้งชื่อ segment จากตารางโปรไฟล์ (ค่ากลาง/เปอร์เซ็นไทล์):

Premium Star — ราคาแพง ขายดี (เจอไม่บ่อยแต่ทรงพลัง)

Premium Niche — ราคาแพง ยอดขายกลาง/น้อย

Mass-Market Hero — ราคาต่ำ–กลาง ขายดี

Long Tail / Mid — ราคาต่ำ–กลาง ยอดขายกลาง/น้อย

### Business actions :

Premium Star → protect stock, premium bundles, testimonials.

Premium Niche → storytelling, education, targeted audiences.

Mass-Market Hero → volume promos, ads optimization.

Long Tail → prune, re-position, or use as add-ons.

### แนวทางใช้งาน :

Premium Star → ดูแลสต็อก โปรโมชันพรีเมียม รีวิวคุณภาพ

Premium Niche → เล่าเรื่อง/อธิบายคุณค่า ยิงโฆษณาเฉพาะกลุ่ม

Mass-Market Hero → โปรโมชันปริมาณ การตลาดเชิงปริมาณ

Long Tail → พิจารณาตัด/ปรับโพสิชันนิ่ง หรือขายพ่วง

## Tips (quick)



If segments look messy, try excluding the optional ```mg/pack``` features or adjust k.

If Thai labels render oddly on plots, install a Thai font in Colab.



ถ้ากลุ่มดูปนๆ ลองตัด ```mg/pack``` ออกหรือปรับจำนวน k

ถ้าฟอนต์ไทยบนกราฟเพี้ยน ให้ติดตั้งฟอนต์ไทยใน Colab
