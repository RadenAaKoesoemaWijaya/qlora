# 📚 Dokumen Arsitektur QLoRA Fine-tuning Platform - Versi Deskriptif

Dokumen ini menyajikan arsitektur platform QLoRA Fine-tuning dalam bentuk uraian naratif komprehensif yang menjelaskan alur kerja, interaksi antar komponen, dan desain sistem secara mendetail.

---

## 1. Gambaran Umum Sistem

Platform QLoRA Fine-tuning adalah sebuah aplikasi web-based yang dirancang untuk melakukan fine-tuning pada model Large Language Model (LLM) menggunakan berbagai metode parameter-efficient seperti QLoRA, DoRA, IA³, VeRA, LoRA+, AdaLoRA, dan OFT. Sistem ini dibangun dengan arsitektur layered yang terdiri dari lima lapisan utama: Client Layer, API Gateway Layer, Business Logic Layer, Data Persistence Layer, dan Infrastructure Layer.

Arsitektur ini dirancang untuk mendukung skalabilitas horizontal, keamanan enterprise-grade, dan kemudahan pemeliharaan. Setiap lapisan memiliki tanggung jawab yang jelas terpisah, memungkinkan pengembangan, testing, dan deployment yang independen antar komponen.

---

## 2. Lapisan Client dan Akses Pengguna

Pengguna dapat mengakses platform melalui berbagai antarmuka yang telah disediakan. Antarmuka utama adalah aplikasi web yang dibangun menggunakan React dengan TypeScript, menyediakan user experience yang responsif dan interaktif. Selain itu, platform juga mendukung akses melalui command line interface (CLI) berbasis Python untuk pengguna yang memerlukan otomatisasi, serta API client menggunakan cURL atau bahasa pemrograman lainnya untuk integrasi programmatic.

Semua akses client ini berkomunikasi dengan backend melalui protokol HTTP/HTTPS dengan format data JSON. Untuk fitur real-time seperti monitoring training progress, sistem menggunakan WebSocket yang memungkinkan push notification dari server ke client tanpa perlu polling berulang.

---

## 3. API Gateway dan Security Middleware

Lapisan API Gateway dibangun menggunakan FastAPI, framework Python modern yang mendukung asynchronous programming. Gateway ini berfungsi sebagai pintu masuk tunggal untuk semua request ke sistem dan dilengkapi dengan serangkaian middleware security.

Pertama, CORS middleware memvalidasi origin header dari setiap request dan memastikan hanya domain yang tercantum dalam whitelist yang diizinkan mengakses API. Kedua, rate limiting berbasis Redis membatasi jumlah request per IP address atau API key dalam periode tertentu, mencegah abuse dan distributed denial of service attacks.

Ketiga, JWT authentication middleware memverifikasi bearer token pada setiap request yang memerlukan autentikasi. Token ini divalidasi secara kriptografis dan dicek expiration time-nya. Keempat, structured logging middleware mencatat setiap request dengan detail seperti timestamp, correlation ID, user ID, endpoint, dan response time untuk keperluan audit dan debugging.

---

## 4. Lapisan Business Logic

Lapisan business logic adalah inti dari platform yang menangani logika bisnis utama. Komponen utama dalam lapisan ini adalah Training Engine Factory yang mengimplementasikan pola desain Factory untuk membuat instance training engine yang sesuai dengan metode yang dipilih pengguna.

Training Engine Factory mendukung tujuh metode fine-tuning berbeda. Setiap metode memiliki engine spesifik yang mengimplementasikan interface BaseTrainingEngine, memastikan konsistensi dalam cara kerja meskipun implementasi internalnya berbeda. Ketika pengguna memilih metode DoRA misalnya, factory akan membuat instance DoRATrainingEngine dengan konfigurasi yang sesuai.

Selain training engines, lapisan ini juga mencakup Data Processor yang menangani parsing dan validasi berbagai format dataset seperti JSON, JSONL, CSV, TXT, Parquet, dan XLSX. Data Processor menggunakan pendekatan streaming untuk file besar guna mengoptimalkan memory usage.

GPU Manager adalah komponen lain yang bertanggung jawab untuk monitoring kesehatan GPU, memilih GPU optimal untuk training job baru, dan mengalokasikan serta membebaskan resource GPU. Komponen ini terintegrasi dengan NVIDIA Management Library untuk mendapatkan informasi real-time tentang GPU utilization, memory usage, dan temperature.

---

## 5. Alur Kerja Training Job

Ketika pengguna memulai training job, serangkaian proses terjadi secara berurutan namun terintegrasi. Pertama, sistem melakukan validasi input menggunakan sembilan validator yang telah ditentukan untuk memastikan parameter seperti LoRA rank berada dalam rentang 1-1024, learning rate antara 1e-6 hingga 1e-2, dan metode training adalah salah satu dari whitelist yang diizinkan.

Setelah validasi berhasil, job baru dibuat dengan status INITIALIZING dan disimpan ke dalam database MongoDB. Sistem kemudian memeriksa ketersediaan GPU melalui GPU Manager dan mengalokasikan GPU yang paling sesuai berdasarkan kriteria seperti memory tersedia, utilization rendah, dan kesehatan device.

Selanjutnya, sistem memuat dataset dari file system. Proses ini dilakukan secara asynchronous dengan timeout lima menit untuk mencegah hang pada file besar. Dataset divalidasi untuk memastikan format yang benar dan keberadaan field-field yang diperlukan seperti instruction dan output.

Training engine kemudian diinisialisasi melalui factory pattern. Engine ini memuat model base dari Hugging Face atau path lokal, mengkonfigurasi quantization ke 4-bit menggunakan BitsAndBytesConfig, dan menyiapkan PEFT configuration sesuai metode yang dipilih.

Training dimulai dengan loop epoch yang mengiterasi seluruh dataset beberapa kali sesuai konfigurasi num_epochs. Selama training, progress tracker secara periodik mengupdate status job di database dan mengirim update real-time ke client melalui WebSocket. Update ini mencakup informasi seperti epoch saat ini, loss value, learning rate, dan estimasi waktu selesai.

Pada akhir setiap epoch, jika konfigurasi mengizinkan, sistem menyimpan checkpoint yang berisi state model yang dapat digunakan untuk resume training atau inference nantinya. Checkpoint ini disimpan ke file system dan metadata-nya tercatat di database.

Setelah training selesai, baik sukses maupun gagal, sistem melakukan cleanup resource. Ini mencakup pemanggilan method cleanup() pada training engine yang menghapus referensi model dari memory, memaksa garbage collection, dan mengosongkan CUDA cache. GPU kemudian dilepas dari job dan tersedia untuk job berikutnya. Status job diupdate ke COMPLETED atau FAILED sesuai outcome, dan pengguna menerima notifikasi final.

---

## 6. Sistem Keamanan Multi-Lapisan

Keamanan adalah aspek fundamental dalam platform ini, diimplementasikan dalam pendekatan defense-in-depth dengan tujuh lapisan validasi. Lapisan pertama adalah validasi path yang menggunakan fungsi validate_dataset_path() untuk mencegah path traversal attacks. Fungsi ini mengkonversi relative path ke absolute path dan memverifikasi bahwa path berada dalam direktori datasets yang diizinkan, menolak setiap upaya mengakses file di luar direktori tersebut.

Lapisan kedua adalah input sanitization yang membersihkan filename dari karakter berbahaya seperti path separators dan null bytes. Sistem juga menghapus leading dots untuk mencegah hidden file access dan membatasi panjang filename maksimal 255 karakter.

Lapisan ketiga adalah role-based access control yang membatasi operasi berdasarkan peran pengguna. Admin memiliki akses penuh ke semua fitur, trainer dapat membuat dan mengelola training jobs serta datasets, sementara viewer hanya memiliki akses read-only.

Lapisan keempat adalah JWT token validation yang memastikan setiap request membawa token yang valid, belum expired, dan ditandatangani dengan benar. Payload token berisi informasi user ID, role, dan permissions yang digunakan untuk authorization decisions.

Lapisan kelima adalah model ID validation yang memverifikasi format Hugging Face model ID sesuai pola username slash model-name. Ini mencegah injection berbahaya melalui model ID yang tidak valid.

Lapisan keenam adalah API key validation untuk programmatic access, memastikan key memiliki panjang minimal 32 karakter dan hanya mengandung printable characters.

Lapisan ketujuh adalah comprehensive training config validation menggunakan Pydantic validators. Sembilan validator memeriksa tipe data, rentang nilai, dan format dari setiap parameter training, menyediakan error messages yang spesifik dan actionable ketika validasi gagal.

---

## 7. Optimasi Performa dan Caching

Platform mengimplementasikan berbagai strategi optimasi performa untuk menangani beban tinggi dan meningkatkan responsiveness. Sistem caching adalah komponen kunci yang menggunakan Redis sebagai distributed cache utama dengan memory cache sebagai fallback ketika Redis tidak tersedia.

Cache manager menyediakan decorator cache_result yang dapat diterapkan pada fungsi-fungsi expensive untuk menyimpan hasil komputasi selama periode tertentu, biasanya 300 detik. Ketika fungsi dipanggil dengan argumen yang sama, hasil dari cache langsung dikembalikan tanpa perlu melakukan komputasi ulang. Ini sangat efektif untuk endpoint seperti daftar training methods dan config schemas yang relatif statis.

Optimasi lain adalah penggunaan asynchronous file processing dengan library aiofiles. Berbeda dengan operasi file blocking tradisional, aiofiles memungkinkan server untuk melanjutkan melayani request lain saat menunggu operasi disk selesai. Ini meningkatkan throughput server secara signifikan.

Untuk database queries, sistem menggunakan MongoDB aggregation pipelines untuk menggabungkan multiple count operations menjadi single query yang efisien. Dashboard stats yang sebelumnya memerlukan lima query terpisah kini dapat diperoleh dalam satu aggregation query yang 60 persen lebih cepat.

Timeout handling adalah aspek penting dalam stability. Setiap operasi yang berpotensi blocking seperti dataset processing diberi timeout lima menit. Jika operasi melebihi batas waktu, sistem secara graceful menangani error, mengupdate job status ke FAILED dengan pesan timeout, dan melakukan cleanup resource.

---

## 8. Manajemen Memori dan Resource Cleanup

Manajemen memori GPU adalah kritis dalam training LLM karena besarnya model. Platform mengimplementasikan mekanisme cleanup yang komprehensif melalui method cleanup() pada BaseTrainingEngine. Method ini menghapus referensi model, trainer, dan tokenizer secara eksplisit, memaksa garbage collection menggunakan gc.collect(), dan mengosongkan CUDA cache dengan torch.cuda.empty_cache() serta melakukan torch.cuda.synchronize() untuk memastikan semua operasi GPU selesai.

Cleanup ini dipanggil dalam blok finally setiap training job, memastikan resource selalu dibebaskan bahkan ketika exception terjadi. Destructor __del__ juga mengimplementasikan fallback cleanup yang mencoba melakukan async cleanup jika event loop sedang berjalan, atau sync cleanup jika tidak.

---

## 9. Sistem Monitoring dan Observabilitas

Observabilitas adalah aspek penting untuk production system. Platform mengintegrasikan Prometheus untuk metrics collection dan Grafana untuk visualization. Metrics yang dikumpulkan mencakup training jobs total, active jobs, GPU utilization, request duration, error rates, dan custom business metrics.

Structured logging menyediakan konteks lengkap untuk setiap event dalam sistem. Setiap log entry mencakup timestamp dengan timezone, level severity, correlation ID untuk request tracing, user ID, job ID, dan metadata operasional lainnya. Log dikirim ke stdout untuk collection oleh logging infrastructure.

Real-time monitoring dashboard menggunakan WebSocket untuk push updates dari server ke client. Ini memungkinkan pengguna melihat progress training secara live tanpa perlu merefresh browser. WebSocket server menerbitkan events seperti epoch completion, loss updates, dan status changes ke channel yang sesuai, dan client yang subscribed menerima updates tersebut secara real-time.

---

## 10. Model Data dan Relasi

Data dalam sistem disimpan dalam MongoDB dengan enam koleksi utama. Koleksi users menyimpan informasi autentikasi dan profil pengguna dengan password yang dihash. Koleksi datasets menyimpan metadata tentang file dataset yang diupload pengguna termasuk path file, tipe format, jumlah baris, dan status validasi.

Koleksi training_jobs adalah inti dari tracking training activities, menyimpan informasi seperti model yang digunakan, dataset ID, status job saat ini, progress persentase, epoch saat ini, total epoch, loss value terkini, learning rate, dan timestamp mulai serta selesai. Setiap job terhubung ke satu user dan satu dataset.

Koleksi training_metrics menyimpan time-series data dari training process, mencatat loss, learning rate, dan step untuk setiap titik waktu selama training. Ini memungkinkan analisis detail dan plotting grafik loss curve.

Koleksi checkpoints menyimpan informasi tentang saved model states, termasuk epoch dan step ketika checkpoint dibuat, path file, dan ukuran file. Checkpoint terhubung ke training job melalui job ID.

Koleksi evaluations menyimpan hasil evaluasi model menggunakan berbagai metrics seperti accuracy, perplexity, F1 score, BERTScore, ROUGE-L, dan BLEU score. Evaluation terhubung ke checkpoint atau model yang dievaluasi.

---

## 11. Deployment dan Infrastruktur

Platform dideploy menggunakan Docker Compose dengan arsitektur multi-container. Container qlora-backend menjalankan FastAPI application dengan Uvicorn server pada port 8000. Container ini memiliki akses ke GPU melalui NVIDIA Docker Runtime dan terhubung ke container MongoDB dan Redis.

Container qlora-frontend menjalankan React SPA yang diserve oleh Nginx pada port 80. Nginx juga berfungsi sebagai reverse proxy untuk backend API dan menyediakan static file serving. Container ini bergantung pada backend untuk API calls.

Container qlora-mongodb menyediakan database MongoDB versi 6.0 dengan persistent storage melalui Docker volume. Data disimpan di luar container sehingga tidak hilang ketika container direstart.

Container qlora-redis menyediakan caching dan session storage dengan Redis versi 7. Container ini juga menggunakan persistent volume untuk data durability.

Container qlora-prometheus mengumpulkan metrics dari backend dan system exporters, menyimpannya dalam time-series database internal. Container qlora-grafana menyediakan visualization interface untuk metrics yang dikumpulkan Prometheus, dengan pre-configured dashboards untuk system dan application monitoring.

Semua container berkomunikasi melalui Docker network bridge yang diberi nama qlora-net, memungkinkan service discovery dan komunikasi antar container menggunakan nama service.

---

## 12. Alur Error Handling dan Recovery

Ketika error terjadi dalam sistem, alur recovery yang terstruktur diaktifkan. Pertama, error diklasifikasikan ke dalam kategori seperti ValidationError, TimeoutError, GPUOutOfMemoryError, DatabaseError, atau TrainingError. Klasifikasi ini menentukan strategi recovery yang sesuai.

Untuk error yang dapat retry seperti timeout, sistem mencoba operasi ulang dengan exponential backoff hingga maksimal tiga kali. Jika retry berhasil, operasi dilanjutkan normal. Jika retry gagal, job diberi status FAILED.

Untuk GPU out of memory error, sistem mencoba recovery dengan mengurangi batch size secara otomatis dan me-restart training dengan parameter yang lebih konservatif. Ini memungkinkan training berhasil selesai meskipun resource GPU terbatas.

Untuk error fatal seperti validation error yang menandakan input tidak valid, sistem segera menghentikan proses, melakukan cleanup resource, mengupdate status job ke FAILED dengan pesan error yang detail, dan mengembalikan error response ke client dengan informasi yang cukup untuk debugging.

Semua error dicatat dalam structured log dengan stack trace dan context lengkap, memudahkan post-mortem analysis dan debugging oleh developer.

---

## 13. Testing dan Quality Assurance

Platform memiliki dua suite test komprehensif. Security test suite mencakup 25 test case yang memverifikasi path traversal protection, input validation, filename sanitization, model ID validation, dan API key validation. Test ini memastikan sistem tahan terhadap berbagai serangan keamanan umum.

Performance test suite mencakup test untuk caching functionality, async file processing, concurrent operations, dan memory usage. Test ini memastikan sistem dapat menangani load tinggi dengan tetap responsif dan tidak mengalami memory leaks.

---

## 14. Keunggulan Arsitektural

Arsitektur platform ini menawarkan beberapa keunggulan signifikan. Desain layered memungkinkan perubahan pada satu lapisan tanpa mempengaruhi lapisan lain, meningkatkan maintainability dan memudahkan pengembangan fitur baru. Penggunaan factory pattern untuk training engines membuat sistem sangat extensible, memungkinkan penambahan metode fine-tuning baru dengan mudah tanpa mengubah code yang ada.

Pendekatan asynchronous programming dengan FastAPI dan asyncio memungkinkan server menangani banyak concurrent connections dengan efisiensi tinggi. Integrasi caching mengurangi beban database dan meningkatkan response time untuk endpoint yang sering diakses. Security hardening dengan tujuh lapisan validasi menyediakan defense-in-depth yang kuat melawan berbagai serangan.

Monitoring dan observability yang komprehensif dengan Prometheus dan Grafana memungkinkan operations team untuk mendeteksi dan diagnose masalah secara proaktif. Deployment containerized dengan Docker memastikan environment konsisten antara development, testing, dan production, mengeliminasi masalah yang disebabkan oleh environment differences.

---

Dengan arsitektur ini, platform QLoRA Fine-tuning siap untuk deployment production dengan performa tinggi, keamanan kuat, dan kemudahan operasional.
