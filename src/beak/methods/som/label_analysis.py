import numpy as np
import pandas as pd
import itertools

def write_som_label_data(workdir, outgeofile, annot_data, annot_strings):
        
        # Set column names based on whether 'outgeofile' is present
        if outgeofile is not None:
            column_names = ["ID", "label", "som_x", "som_y", "X", "Y"]
        else:
            column_names = ["ID", "label", "som_x", "som_y"]

        # Create DataFrame with specified column names
        labels = pd.DataFrame(annot_data, columns=column_names)

        # Save the DataFrame to a text file
        labels.to_csv(workdir + '/labels_flat.txt', sep='\t', index=False)

        #list_grouped=list(annot_strings.items())
        array_grouped=np.array(list(annot_strings.items()))#dict to list and list to np array
        for i in range(0,len(array_grouped)):
            array_grouped[i][1]= array_grouped[i][1][(array_grouped[i][1].find(":")+1):len(array_grouped[i][1])]      #  ": "+ ','.join(annot_strings[str(i)])

        # Save the array to a text file
        np.savetxt(workdir + '/labels_grouped.txt', array_grouped, delimiter='\t', fmt='%s')

        return labels


def write_bmu_cluster_label_data(workdir, cluster_array, geo_data, labels):
        #---------------------------------------------------
        ##--- Identify all geo points in label-bearing BMU

        # Identify rows with noDataValue and delete
        rows_to_delete = np.any(np.isnan(geo_data), axis=1)
        geo_data = geo_data[~rows_to_delete]

        # Initialize an empty dictionary to store BMUs with labels
        bmu_labeled = {}

        # Iterate over each row in labels DataFrame
        for index, row in labels.iterrows():
            som_x, som_y = int(row['som_x']), int(row['som_y'])
            bmu_key = (som_x, som_y)
            label = row['label']

            # If the BMU key is not in bmu_labeled, add it with an empty list for labels and geo points
            if bmu_key not in bmu_labeled:
                
                # get cluster 
                if cluster_array is not None:
                    cluster = int(cluster_array[som_y][som_x])
                else:
                    cluster = 0

                bmu_labeled[bmu_key] = {'cluster': cluster, 'labels': [], 'geo_xy': []}

                # Find matching SOM coordinates in geo_data and append geo points to corresponding BMU
                matching_rows = geo_data[(geo_data[:, 2] == som_x) & (geo_data[:, 3] == som_y)]
                for matching_row in matching_rows:
                    bmu_labeled[bmu_key]['geo_xy'].append((matching_row[0], matching_row[1]))

            # Append label to corresponding BMU
            bmu_labeled[bmu_key]['labels'].append(label)


        #---------------------------------------------------
        #--- Extract all geo points with som_x, som_y from bmu_labeled and count labels per cluster
        geo_label_bmu = []

        # Create an empty dictionary to store the label counts for each cluster
        cluster_label_counts = {}

        # Iterate over each BMU in bmu_labeled
        for bmu_key, bmu_info in bmu_labeled.items():
            # Extract BMU coordinates
            som_x, som_y = bmu_key

            # Extract cluster information
            cluster = bmu_info['cluster']

            # Extract labels and geo points
            labels = bmu_info['labels']
            #geo_xy = bmu_info['geo_xy']

            # Count the number of labels
            label_count = len(labels)

            # Update the label count for the current cluster
            if cluster in cluster_label_counts:
                cluster_label_counts[cluster] += label_count
            else:
                cluster_label_counts[cluster] = label_count
            
            for geo_point in bmu_info['geo_xy']:
                geo_x, geo_y = geo_point
                geo_label_bmu.append([geo_x, geo_y, int(som_x), int(som_y), int(label_count), int(cluster)])

        for idx, row in enumerate(geo_label_bmu):
            cluster = row[5]
            cluster_label_count = cluster_label_counts[cluster]
            geo_label_bmu[idx].append(int(cluster_label_count))

        # Convert geo_label_bmu list to a DataFrame
        geo_label_df = pd.DataFrame(geo_label_bmu, columns=['X', 'Y', 'SOM_X', 'SOM_Y', 'Label_Count', 'Cluster', 'Cluster_Label_Count'])

        # Convert the DataFrame back to a NumPy array
        bmu_geo_label_data = geo_label_df.to_numpy()

        # Add BMU ID column
        sorted_bmus = sorted(bmu_labeled.keys())
        bmu_id_map = {bmu: idx for idx, bmu in enumerate(sorted_bmus)}
        bmu_geo_label_data_with_id = np.empty((bmu_geo_label_data.shape[0], bmu_geo_label_data.shape[1] + 1), dtype=float)
        for idx, row in enumerate(bmu_geo_label_data):
            bmu_key = (row[2], row[3])
            bmu_id = bmu_id_map[bmu_key]
            bmu_geo_label_data_with_id[idx] = np.concatenate(([bmu_id], row))

        # Sort the array by the added BMU ID column
        sorted_indices = np.lexsort((bmu_geo_label_data_with_id[:, 0],))

        # Reorder the array based on the sorted indices
        sorted_array = bmu_geo_label_data_with_id[sorted_indices]

        # Save the array to a text file
        np.savetxt(workdir + "/" + "geo_labeled_bmu.txt", sorted_array, delimiter='\t', fmt='%d\t%f\t%f\t%d\t%d\t%d\t%d\t%d', header='bmu_id\tgeo_x\tgeo_y\tsom_x\tsom_y\tbmu_label_count\tcluster\tcluster_label_count')

        #---
        # Convert the dictionary to a list of tuples
        cluster_label_counts_list = [(cluster, count) for cluster, count in cluster_label_counts.items()]
        cluster_label_counts_list.sort(key=lambda x: x[0], reverse=False)

        # Convert the list to a numpy array
        cluster_label_counts_array = np.array(cluster_label_counts_list)

        # Save the array to a text file
        np.savetxt(workdir + "/" + "cluster_label_counts.txt", cluster_label_counts_array, delimiter='\t', fmt='%s', header='Cluster\tLabel_Count')



def read_geo_labeled_bmu(file_path):
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    
    bmu_id = data[:, 0]
    geo_x = data[:, 1]
    geo_y = data[:, 2]
    som_x = data[:, 3].astype(int)
    som_y = data[:, 4].astype(int)
    label_count = data[:, 5].astype(int)
    cluster = data[:, 6].astype(int)
    cluster_label_counts = data[:, 7].astype(int)
    
    return {
        'bmu_id': bmu_id,
        'X':geo_x, 
        'Y':geo_y, 
        'SOM X':som_x, 
        'SOM Y':som_y, 
        'BMU label count':label_count, 
        'cluster':cluster, 
        'cluster label count':cluster_label_counts
        }


def get_label_annotation_data(som_dict, geo_data, outgeofile, noDataValue):
    
    somx = som_dict['n_columns']
    somy = som_dict['n_rows']
    
    annot_ticks = np.empty([somx, somy], dtype='<U32')
    annot_ticks.fill("")

    annot_strings = {}
    annot_data = []
    index_label = []
    index_nolabel = []

    annot_strings_for_dict={}

    # get label index in data_file
    #with open(data_file, encoding='utf-8-sig') as fh:
    #    header_line = fh.readline()
    #colnames = header_line.split() if outgeofile is not None else header_line.split("\t")
    labelIndex = geo_data.shape[1] - 1
    #labelIndex = colnames.index('label')

    # "best matching unit" in som space for each data containing grid point in geo space
    bmus = som_dict["bmus"]
    
    data_label=geo_data[:, labelIndex].astype(str)
    unique_labels = set(data_label) - {'0.0', '', "nan", "NA", "NULL", "Null", "NoData", noDataValue}
    if len(unique_labels) == 1:
        common_label = next(iter(unique_labels))
        # If all labels are the same, store the count for the common label
        for i in range(len(data_label)):
            if data_label[i] == common_label:
                index_label.append(i)
                tick = annot_ticks[bmus[i][0]][bmus[i][1]]
                counter = len(annot_strings) + 1
                if tick == '':
                    annot_ticks[bmus[i][0]][bmus[i][1]] = str(counter)
                    annot_strings[str(counter)] = {common_label: 1}  # Store label and count as a dictionary
                    annot_strings_for_dict[str(counter)] = {common_label: 1}
                    annot_data.append([f"{counter}", f"{common_label}", f"{bmus[i][0]}", f"{bmus[i][1]}",
                                       f"{geo_data[i][0]}" if outgeofile is not None else None, f"{geo_data[i][1]}" if outgeofile is not None else None])
                else:
                    annot_strings[tick][common_label] += 1
                    annot_strings_for_dict[tick][common_label] += 1
                    annot_data.append([f"{tick}", f"{common_label}", f"{bmus[i][0]}", f"{bmus[i][1]}",
                                       f"{geo_data[i][0]}" if outgeofile is not None else None, f"{geo_data[i][1]}" if outgeofile is not None else None])
            else:
                index_nolabel.append(i)
    else:
        # If there are different labels, get a list of labels in each BMU
        for i in range(0, len(data_label)):
            if data_label[i] not in ['0.0', '', "nan", "NA", "NULL", "Null", "NoData", noDataValue]:
                label = int(float(data_label[i]))
                index_label.append(i)
                tick = annot_ticks[bmus[i][0]][bmus[i][1]]
                counter = len(annot_strings) + 1
                if tick == '':
                    annot_ticks[bmus[i][0]][bmus[i][1]] = str(counter)
                    annot_strings[str(counter)] = {label: 1}  # Store label and count as a dictionary
                    annot_strings_for_dict[str(counter)] = {label: 1}
                    annot_data.append([f"{counter}", f"{label}", f"{bmus[i][0]}", f"{bmus[i][1]}",
                                       f"{geo_data[i][0]}" if outgeofile is not None else None, f"{geo_data[i][1]}" if outgeofile is not None else None])
                else:
                                # Check if the label already exists in annot_strings[tick]
                    if label in annot_strings[tick]:
                        annot_strings[tick][label] += 1
                        annot_strings_for_dict[tick][label] += 1
                    else:
                        annot_strings[tick][label] = 1  # Initialize count if label doesn't exist
                        annot_strings_for_dict[tick][label] = 1
                    annot_data.append([f"{tick}", f"{label}", f"{bmus[i][0]}", f"{bmus[i][1]}",
                                       f"{geo_data[i][0]}" if outgeofile is not None else None, f"{geo_data[i][1]}" if outgeofile is not None else None])
            else:
                index_nolabel.append(i)
    
    # Sort annot_strings by key
    for i in range(1, len(annot_strings) + 1):
        annot_strings[str(i)] = {k: v for k, v in sorted(annot_strings[str(i)].items())}

    # Merge duplicates within a labeling group
    for i, j in itertools.combinations(range(1, counter + 1), 2):

        if annot_strings.get(str(i)) == annot_strings.get(str(j)):

            #remove duplicate entry from dictionary 'annot_strings'
            annot_strings.pop(str(j), None)

            #go through all indices of 2D array 'annot_ticks' and replace value if equal to duplicate entry j
            for a, b in itertools.product(range(len(annot_ticks)), range(len(annot_ticks[0]))):
                if annot_ticks[a][b] == str(i) or annot_ticks[a][b] == str(j):
                    if len(unique_labels) == 1:
                        annot_ticks[a][b] = 'x' + str(i)
                    else:
                        annot_ticks[a][b] = str(i)

    if len(unique_labels) == 1:
        # Check if 'x' is included in all annot_ticks elements, if not (when no duplicates), add it
        for a in range(len(annot_ticks)):
            for b in range(len(annot_ticks[a])):
                if 'x' not in annot_ticks[a][b] and not annot_ticks[a][b]=='':
                    annot_ticks[a][b] = 'x' + annot_ticks[a][b]

    # set new index numbers
    if len(unique_labels) == 1:
        # Sort annot_strings by label count
        sorted_indices = sorted(annot_strings.keys(), key=lambda x: sum(annot_strings[x].values()))
        new_indices = {}
        for index in sorted_indices:
            count = sum(annot_strings[index].values())
            new_indices[index] = str(count)
            for a in range(len(annot_ticks)):
                for b in range(len(annot_ticks[a])):
                    if annot_ticks[a][b] == 'x' + index:
                        annot_ticks[a][b] = str(count)


        # Update annot_strings with the new indices
        #annot_strings = {new_index[index]: labels for index, labels in annot_strings.items()}
        #annot_strings = {new_index: labels for new_index, labels in sorted(annot_strings.items(), key=lambda x: int(x[0]))}
        annot_strings = {new_index: annot_strings[index] for index, new_index in new_indices.items()}

        # Sort annot_strings by ascending "count"
        annot_strings = {k: v for k, v in sorted(annot_strings.items(), key=lambda item: sum(item[1].values()))}

    else:
        counter = 0
        for i in range(1, len(annot_strings_for_dict) + 1):
            if str(i) in annot_strings:
                counter += 1
                annot_strings[str(counter)] = annot_strings.pop(str(i))
                for a, b in itertools.product(range(len(annot_ticks)), range(len(annot_ticks[0]))):
                    if annot_ticks[a][b] == str(i):
                        annot_ticks[a][b] = str(counter)

    for key in annot_strings:
        #annot_strings[str(key)] = f"{key}: {','.join([f'{label}({count})' for label, count in annot_strings[str(key)].items()])}"
        annot_strings[key] = f"{key}: {','.join([f'{label}({count})' for label, count in annot_strings[key].items()])}"
    
    return annot_ticks, annot_strings, annot_data, index_label, index_nolabel
