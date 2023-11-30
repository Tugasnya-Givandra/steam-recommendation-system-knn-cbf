# Steam Recommender System 😎

## Quickstart untuk para dev
### Struktur kode projek

```
.
├── Home.py                         # Entry untuk
├── README.md                       # We are here
├── data_preprocessing.ipynb        # kodingan buat ngolah dataset 
│                                   # kaggle biar sesuai ama model
├── datasets
│   ├── archive                     # dataset asli kaggle
│   │   ├── games.csv
│   │   ├── games_metadata.json
│   │   ├── recommendations.csv
│   │   └── users.csv
│   ├── clean_games.pkl             # hasil olahan isinya data game
│   │                               # dan tags semua game dalam bentuk
│   │                               # dummy (0 1)
│   ├── id_to_title.pkl             # helper untuk ngubah app_id jadi title
│   └── title_to_id.pkl             # helper untuk ngubah title jadi app_id
├── pages                           # folder untuk sub page
│   ├── About.py                    # halaman untuk nanti bagian penjelasan
│   │                               # algoritmanya (buat presentasi)
│   ├── Lihat Jeroan.py             # Buat debug, nampilin data yang di process
│   ├── Results.py                  # Hasil rekomendasi diredirect dari home
├── requirements.txt                # keperluan modul untuk run
└── utils                           # yang mungkin dibutuhkan pages
    ├── Model.py                    # model KNNCBF
    ├── home_utils.py
    └── utils.py
```

### Setup pertama kali
Saya sarankan menggunakan virtualenv untuk projek ini

```
pip installl -r requirements.txt
```

Anda redi untuk ngerjain


### Alur data
1. Dari home user bakal pilih x game, user bisa milih jenisnya suka atau tidak
2. User lalu tekan button `get recommendation`, list game ini akan
disimpan di session yang nantinya bakal diolah sama model. Setelah selesai, redirect ke results
3. Sebelum render, olah hasil pilihan user ke KNNCBF
4. Setelah sampai di results, tampilin 10 match terbaik dari rekomendasi
5. User bisa nyentang nyentang mana yang relevan dari yang dia masukin dan tidak relevan
5. Habis tu liat metrik performa model