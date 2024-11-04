module Np = Np.Numpy;;

(* Layout of the data for housing in california *)
type housing_data = {
  longitude : float;
  latitude : float;
  median_house_value : float;
}

let _print_csv (home_data: housing_data array) =  Array.iter (fun record ->
    Printf.printf "Longitude: %f, Latitude: %f, Median House Value: %f\n"
      record.longitude record.latitude record.median_house_value
  ) home_data
;;


(* Parse the row, removing unused data *)
let parse_row row =
    {
      longitude = float_of_string (List.nth row 0);
      latitude = float_of_string (List.nth row 1);
      median_house_value = float_of_string (List.nth row 8);
    }
;;

(* Load the CSV, translating it to our data structure, and translating it to `Array` so it can be used by Numpy *)
let load_csv (filename : string) : housing_data array =
    let csv_data = Csv.load filename in
    (* Skip the header and parse each row into housing_data *)
    List.tl csv_data |> List.map parse_row |> Array.of_list
let home_data = load_csv ("housing.csv");;

(* Load the data into a matrix *)
let y = Np.matrixf (Array.map (fun d -> [|d.median_house_value|]) home_data);;
print_string "preparing X\n";;
let x = Np.matrixf (Array.map (fun d -> [|d.latitude; d.longitude|]) home_data);;

print_string "Splitting training data\n";;
(* Ignore the warning about being non-exhaustive, as this is correct. *)
let [@ocaml.warning "-8"][x_train; x_test; _y_train; _y_test] = Sklearn.Model_selection.train_test_split [x; y] ~test_size:(`F 0.33) ~random_state:0;;

print_string "Normalize training data\n";;
let (x_traing_normal, _y) = Sklearn.Preprocessing.normalize ~x:x_train ();;
print_string "Normalize testing data\n";;
let (_x_test_normal, _y) = Sklearn.Preprocessing.normalize ~x:x_test ();;

let x_traing_normal = Np.reshape ~newshape:[-1; 1](x_traing_normal);;

print_string "Create kmeans\n";;
let kmeans = Sklearn.Cluster.KMeans.create ~n_clusters:2 ~n_init:(2) ();;
print_string "Fit data\n";;
let _ = Sklearn.Cluster.KMeans.fit ~x:x_traing_normal (kmeans);;
print_string "Labels\n";;
let (labels) = Sklearn.Cluster.KMeans.labels_ (kmeans);;
Sklearn.Metrics.silhouette_samples  ~metric:(`S "euclidean") ~x:(`Arr x_traing_normal) ~labels ();;
print_string "print labels\n";;
Np.Obj.print labels;;

print_string "H";;