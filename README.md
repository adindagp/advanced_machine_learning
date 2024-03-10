# advanced_machine_learning

Advanced Machine Learning Project
K Nearest Neighbor
Adinda Gita P
Penjelasan secara umum
K Nearest neighborhood merupakan model yang biasanya digunakan untuk menyelesaikan masalah klasifikasi. Secara umum model ini akan mengklasifikasikan data baru berdasarkan jaraknya ke kelompok data training. Penghitungan jarak bisa dilakukan dengan beberapa metode, salah satunya Euclidian distance. Penentuan klasifikasi suatu titik data baru akan didasarkan dari jenis kelompok k data terdekat, di mana k merupakan parameter yang ditentukan oleh pengguna dan menjadi objek optimasi. 

Pseudocode
Berikut adalah pseudocode yang dibuat pada project ini

	import numpy as np
	from scipy.spatial import distance as dist

	class NearestNeighbors:
	#1. Pembentukan kelas, parameter, dan parameter bawaan jika tidak diatur oleh pengguna
	class NearestNeighbors:
    	def __init__(self,
                 n_neighbors = 5 ):
        	self.n_neighbors = n_neighbors

    	def fit(self, x, y):
    	# 2. Pembentukan metode fit dalam kelas Nearest Neighbor. X adalah titik data dan Y adalah kelas yang sesuai. Data X 	dan Y diubah menjadi bentuk array
        	self.x = np.array(x)
        	self.y = np.array(y)

    	def predict(self, data_tes):
    	# 3. Metode prediksi
    	Enumerasi untuk menghitung Euclidian distance titik data i ke semua titik data lainnya.
     
        	distances = []
        	for i, x in enumerate(self.x):
		# 4. Kemudian semua hasil perhitungan jarak dan jenis kelas dari titik data akan dimasukan ke dalam list 		bernama distance. Semua jarak akan diurutkan dari yang paling dekat ke titik ke paling jauh Seleksi k-titik 		terdekat dan label kelasnya.
            		dist_val = dist.euclidean(data_tes, x)
            		distances.append((dist_val, self.y[i]))
        	distances.sort()
        	nearest_classes = [item[1] for item in distances[:self.n_neighbors]]
        	predicted_class = max(set(nearest_classes), key=nearest_classes.count)
        	return predicted_class

	#Trial with very simple data
	x_data = [(10,50),(8,60),(7,40),(0,10),(1,0)]
	y_data = [0,0,0,1,1]

	knn = NearestNeighbors(n_neighbors=3)
	knn.fit(x_data, y_data)

	new_point = (5, 30)
	predicted_class = knn.predict(new_point)
	print("Predicted class:", predicted_class)
	
Daftar Pustaka

Neeb, Henry, and Christopher Kurrus. "Distributed K-Nearest Neighbors." June 5, 2016.

Grus, Joel. Data Science from Scratch. O'Reilly Media, 2015.

Navlani, Avinash. "KNN Classification using Scikit-learn." Machine Learning Geek, March 6, 2021. Available at: https://machinelearninggeek.com/knn-classification-using-scikit-learn/

