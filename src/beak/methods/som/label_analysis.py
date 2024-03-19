import numpy as np
import pandas as pd

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
        rows_to_deleted = np.any(np.isnan(geo_data), axis=1)
        geo_data = geo_data[~rows_to_deleted]

        # Initialize an empty dictionary to store BMUs with labels
        bmu_labeled = {}

        # Iterate over each row in labels DataFrame
        for index, row in labels.iterrows():
            som_x, som_y = int(row['som_x']), int(row['som_y'])
            bmu_key = (som_x, som_y)
            label = row['label']
            cluster = int(cluster_array[som_x][som_y])

            # If the BMU key is not in bmu_labeled, add it with an empty list for labels and geo points
            if bmu_key not in bmu_labeled:
                bmu_labeled[bmu_key] = {'cluster': cluster, 'labels': [], 'geo_xy': []}

            # Append label to corresponding BMU
            bmu_labeled[bmu_key]['labels'].append(label)

            # Find matching SOM coordinates in geo_data and append geo points to corresponding BMU
            matching_rows = geo_data[(geo_data[:, 2] == som_x) & (geo_data[:, 3] == som_y)]
            for matching_row in matching_rows:
                bmu_labeled[bmu_key]['geo_xy'].append((matching_row[0], matching_row[1]))


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
                geo_label_bmu.append([geo_x, geo_y, int(som_x), int(som_y), int(label_count), int(cluster), int(cluster_label_counts[cluster])])

        # Convert the list of geo points to a NumPy array
        bmu_geo_label_data = np.array(geo_label_bmu)

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