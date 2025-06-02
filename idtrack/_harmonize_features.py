#!/usr/bin/env python3

import gc
import logging
import os
import pickle
import sys
from functools import cached_property
from typing import Literal, List

import idtrack

import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import tqdm
import warnings


MISSING_VALUES = [
    "na", "NA", "Na", "n/a", "N/A", "n.a.", "N.A.", 
    "na.", "Na.", "NA.", "n-a", "N-A", "Na-", "NA-", "n_a", "N_A", "Na_", "NA_",
    "none", "None", "NONE", "non", "Non", "NON",
    "void", "Void", "VOID",
    "blank", "Blank", "BLANK",
    "omitted", "Omitted", "OMITTED",
    "NaN", "nan", 'Nan', "NAN", 
    "-", "_", "--", "__", "---", "___",
    "null", "Null", "NULL", "nil", "Nil", "NIL", 
    "undefined", "Undefined", "UnDefined", "UNDEFINED", 
    "unassigned", "Unassigned", "UnAssigned", "UNASSIGNED", 
    "unspecified", "Unspecified", "UNSPECIFIED", 
    "", " ", "  ", "   ", "    ", 
    ".", "..", "...", "....", 
    "?", "??", "???", "????",
    "missing", "Missing", "MISSING", 
    "absent", "Absent", "ABSENT", 
    "EMPTY", "empty", "Empty",
    "not available", "Not Available", "NOT AVAILABLE", "notavailable", "NotAvailable", "NOTAVAILABLE",
    "unknown", "Unknown", "UnKnown", "UNKNOWN",
    "not known", "Not Known", "NOT KNOWN", "notknown", "NotKnown", "NOTKNOWN",
    "undisclosed", "Undisclosed", "UNDISCLOSED", 
    "not disclosed", "Not Disclosed", "NOT DISCLOSED", "notdisclosed", "NotDisclosed", "NOTDISCLOSED", 
    "not applicable", "Not Applicable", "NOT APPLICABLE", "notapplicable", "NotApplicable", "NOTAPPLICABLE", 
    "to be filled", "To Be Filled", "TO BE FILLED", 
    "to be determined", "To Be Determined", "TO BE DETERMINED", 
    "tbd", "TBD", "tba", "TBA",
    "n.k.", "N.K.", "n.k", "N.K", "n/k", "N/K", "n/k/a", "N/K/A", 
    "n.d.", "N.D.", "n.d", "N.D", "n/d", "N/D", "n/d/a", "N/D/A", 
    "n.a.p.", "N.A.P.", "n.a.p", "N.A.P", "n/a/p", "N/A/P", "n/a/p/a", "N/A/P/A", 
    "xx", "XX", "x", "X", "xxx", "XXX", "xxxx", "XXXX",
    "N/M", "n/m", "N.M.", "n.m.", "n.m", "N.M",
    "N/C", "n/c", "N.C.", "n.c.", "n.c", "N.C",
    "n.s.", "N.S.", "n.s", "N.S", "n/s", "N/S", "n/s/a", "N/S/A"
]

NA_CELL_TYPE_PLACEHOLDER = "NA"

GENE_NAME_COLUMN = "hgnc"



class HarmonizeFeatures:
    
    nonmatching_but_consistent_suffix = ""
    
    def __init__(
        self,
        project_name,
        data_h5ad_dict,
        project_tmp_dir,
        idtrack_tmp_dir,
        target_ensembl_release,
        last_ensembl_release=110,
        final_database="HGNC Symbol",
        organism_name="homo_sapiens",
        verbose: Literal[0, 1, 2] = 0,
        debugging_variables: bool = False
    ):
        self.project_name = project_name
        self.project_tmp_dir = project_tmp_dir
        self.data_h5ad_dict = data_h5ad_dict

        self.idtrack_tmp_dir = idtrack_tmp_dir
        self.target_ensembl_release = target_ensembl_release
        self.last_ensembl_release = last_ensembl_release
        self.final_database = final_database
        self.organism_name = organism_name
        self.n_datasets = len(self.data_h5ad_dict)
        self.debugging_variables = debugging_variables

        if verbose == 0:
            _logging = logging.ERROR
        elif verbose == 1:
            _logging = logging.WARNING
        elif verbose == 2:
            _logging = logging.INFO
        elif verbose == 3:
            _logging = logging.DEBUG
        else:
            raise ValueError
        self.verbose = verbose

        self.idt = idtrack.API(local_repository=self.idtrack_tmp_dir)
        self.idt.configure_logger(_logging)
        self.idt_initialized = False

        self.dict_n_to_1 = dict()
        self.multiple_ensembl_list = list()
        
        # Important dictionaries for debugging:
        
        # If there is n-to-1 matching in a database and if one of them is query=matching, keep it and remove others
        # for a given group (n), below shows the chosen id (due to its identical as explained), the group and the datasets that has this group
        self.dict_n_to_1_with_query = dict()  
        # this dictionary is ideal to find out the removed ids
        self.dict_n_to_1_with_query_reverse = dict()
        # this is a dictary showing group-to-dataset
        self.dict_n_to_1_without_query = dict()
        # also: 
        # cached_property: self.dict_1_to_not_1
                
        # to include into integrated anndata object.
        
        # sometimes, hgnc id is associated with multiple ensembl ids, the algorithm chooses one of them and creates a list. here you can find the actual possible ensembl ids.
        self.multiple_ensembl_dict = dict()  
        # removed ids from the anndata by `harmonizer` method. the list and the in which datasets it is deleted is kept.
        self.removed_conversion_failed_identifiers = dict()
        # these ids could be removed but kept as they are consistent across all provided datasets.
        self.kept_conversion_failed_identifiers = dict()
        # inconsistent matching
        self.removed_inconsistent_identifier_matching = dict()
        
        self._initialize()

    def _initialize(self): 
        
        for dataset_name, dataset_path in self.data_h5ad_dict.items():
            gene_list = self.extract_source_identifiers_from_anndata(dataset_path)
            
            self.reporter_dict_creator(dataset_name=dataset_name,
                the_dict=self.removed_conversion_failed_identifiers, 
                the_set=set(gene_list) & self.conversion_failed_identifiers)
            
            self.reporter_dict_creator(dataset_name=dataset_name,
                the_dict=self.kept_conversion_failed_identifiers, 
                the_set=set(gene_list) & self.conversion_failed_but_consistent_identifiers)
            
            self.reporter_dict_creator(dataset_name=dataset_name,
                the_dict=self.removed_inconsistent_identifier_matching, 
                the_set=set(gene_list) & self.datataset_conversion_dataframe_issues["gene_names_inconsistency"])
            
            for the_id in gene_list:
                conversion = self.unified_matching_dict[the_id]["matching"]["last_node"]
                if the_id not in self.conversion_failed_but_consistent_identifiers:
                    self.multiple_ensembl_list.append(conversion)

        self.multiple_ensembl_dict = self.create_multiple_ensembl_dict()
        self.datataset_conversion_dataframe_issues
        
    def _initialize_idt(self):
        if not self.idt_initialized:
            organism_formal_name, _ = self.idt.get_ensembl_organism(self.organism_name)
            self.idt.initialize_graph(
                organism_name=organism_formal_name, 
                last_ensembl_release=self.last_ensembl_release
            )
            self.idt.calculate_graph_caches()
            self.idt_initialized = True
            
    def create_multiple_ensembl_dict(self):
        
        r = dict()
        for i in self.multiple_ensembl_list:
            for n, m in i:
                if m not in r:
                    r[m] = set()
                r[m].add(n)
        r = {k: sorted(v) for k, v in r.items()}
    
        return r

    def reporter_dict_creator_helper_reason_finder(self, the_id):
        
        reason = []
        
        matching = self.unified_matching_dict[the_id]
        len_targets = len(matching["matching"]["target_id"])

        if the_id in self.dict_n_to_1:  # in one of the datasets during the integration, this ID was a part of n-to-1.
            reason.append("n-to-1")
        if len_targets == 0:
            reason.append("1-to-0")
        if len_targets > 1:
            reason.append("1-to-n")
        
        return "_".join(reason)

    def reporter_dict_creator(self, the_dict, the_set, dataset_name):
        for i in the_set:
            if i not in the_dict:
                the_dict[i] = {
                    "reason": self.reporter_dict_creator_helper_reason_finder(i),
                    "datasets_containing": []
                }
            the_dict[i]["datasets_containing"].append(dataset_name)

    @cached_property
    def datataset_conversion_dataframe_issues(self):
        
        gene_list = set(self.unified_matching_dict.keys()) - self.conversion_failed_identifiers
        df = self.create_dataset_conversion_dataframe(gene_list=gene_list, initialization_run=True)
        
        # In Shiddar and Wang contains FAM231C, and FAM231B. Very weird genes, IDTrack fails:
        # ENSG00000268674.2	FAM231C	FAM231C
        # ENSG00000268674.2	FAM231B	FAM231B
        # Report and remove from all conversion lists.
        dfa = df.drop_duplicates(subset=['ensembl_gene', self.final_database], keep='first')
        _gene_names_inconsistency = set(dfa[dfa["ensembl_gene"].duplicated(keep=False)][self.final_database].unique())
        gene_names_inconsistency = set(df[df[self.final_database].isin(_gene_names_inconsistency)]["Query ID"].unique())
        
        hgnc_chosen_single_ensembl_dict = dict()
        for final_hgnc_gene_name, df_subset in dfa[dfa[self.final_database].duplicated(keep=False)].groupby(self.final_database):
            # Due to the path, idtrack may choose different ensembl_gene for corresponding HGNC. Here, make a rule for each hgnc. 
            # You can see all possible ensembl ids in 
            chosen_ensembl_id = sorted(df_subset["ensembl_gene"])[0]  # choose first one
            hgnc_chosen_single_ensembl_dict[final_hgnc_gene_name] = chosen_ensembl_id
            
        return {"gene_names_inconsistency": gene_names_inconsistency, "hgnc_chosen_single_ensembl_dict": hgnc_chosen_single_ensembl_dict}

    def create_dataset_conversion_dataframe(self, gene_list, initialization_run):
        
        assert len(self.conversion_failed_identifiers) != 0 and len(self.conversion_failed_but_consistent_identifiers) != 0, "Possible function call order issue!"
        
        new_var_list = list()
        for the_id in gene_list:
            conversion = self.unified_matching_dict[the_id]["matching"]["last_node"]
            
            if the_id in self.conversion_failed_but_consistent_identifiers:
                modified_id = (f"{the_id}{HarmonizeFeatures.nonmatching_but_consistent_suffix}", the_id)
                new_var_list.append(list(modified_id) + [the_id])
                
            # 1-to-n's and 1-to-0s are either removed or kept as consistent ones. 
            # if there is multiple ensembl id matching, then len(conversion) != 1. 
            # it keeps the first one for now, the possible lists are saved self.multiple_ensembl_list in `initialize` method        
            
            elif initialization_run:   
                new_var_list.append(list(conversion[0]) + [the_id])
            else:
                c = list(conversion[0])
                assert len(c) == 2
                if c[1] in self.datataset_conversion_dataframe_issues["hgnc_chosen_single_ensembl_dict"]:
                    c[0] = self.datataset_conversion_dataframe_issues["hgnc_chosen_single_ensembl_dict"][c[1]]
                
                new_var_list.append(c + [the_id])
                
        df = pd.DataFrame(new_var_list, columns=["ensembl_gene", self.final_database, "Query ID"])
        return df

    def feature_harmonizer(self, dataset_name: str):
        
        adata = ad.read_h5ad(self.data_h5ad_dict[dataset_name])
        t0 = adata.n_vars
        remove_bool = ~(
            adata.var.index.isin(self.conversion_failed_identifiers) | \
            adata.var.index.isin(self.datataset_conversion_dataframe_issues["gene_names_inconsistency"])
        )
        adata = adata[:, remove_bool]
        
        df = self.create_dataset_conversion_dataframe(gene_list=adata.var.index, initialization_run=False)
        
        bool_remove = df["ensembl_gene"].duplicated(keep=False).any() or df[self.final_database].duplicated(keep=False).any() 
        assert not bool_remove, "Unexpected duplicated entry!"
        
        df = df.set_index("ensembl_gene")
        resulting_adata = ad.AnnData(X=adata.X, obs=adata.obs, var=df)  # dtype=adata.X.dtype,
        
        t1 = adata.n_vars
        del adata
        gc.collect()
        
        return resulting_adata, t0, t1

    @cached_property
    def dict_1_to_not_1(self):
        result = dict()
        for query, value in self.unified_matching_dict.items():
            if len(value["matching"]["target_id"]) != 1:
                result[query] = value["datasets_containing"]

        return result

    @cached_property
    def conversion_failed_identifiers(self):
        result = set()
        
        for query, datasets_containing in self.dict_1_to_not_1.items():
            # for 1-to-n and 1-to-0, if it is not shared across all the datasets, add remove list.
            if len(datasets_containing) != self.n_datasets:
                result.add(query)
        
        for query, datasets_containing in self.dict_n_to_1.items():
            # for one gene in n (among n-to-1), if it is not shared across all the datasets, add remove list.
            if len(datasets_containing) != self.n_datasets:
                result.add(query)
        
        return result
            
    @cached_property
    def conversion_failed_but_consistent_identifiers(self): 

        _result1 = {i for i, _ in self.dict_n_to_1.items() if i not in self.conversion_failed_identifiers}
        _result2 = {i for i, _ in self.dict_1_to_not_1.items() if i not in self.conversion_failed_identifiers}
        result = set.union(_result1, _result2)
        
        return result

    @cached_property
    def unified_matching_dict(self):
        matching_dict = self.get_idtrack_matchings_for_all_datasets()
        unified_dict = dict()
        
        for dataset_name, matching in matching_dict.items():
            for m in matching:
                if m["query_id"] not in unified_dict:
                    unified_dict[m["query_id"]] = {
                        "matching": m, 
                        "datasets_containing": []
                    }                    
                unified_dict[m["query_id"]]["datasets_containing"].append(dataset_name)

        return unified_dict

    def get_idtrack_matchings_for_all_datasets(self):
        result_dataset = dict()
        for dataset_name, dataset_path in self.data_h5ad_dict.items():
            dataset_pickle = os.path.join(
                self.project_tmp_dir, f"idtrack_result_{self.project_name}_{dataset_name}.pickle"
            )

            switch_inconsistency = None
            if os.path.isfile(dataset_pickle) and os.access(dataset_pickle, os.R_OK):
                with open(dataset_pickle, "rb") as handle:
                    matching_list = pickle.load(handle)
                _saved_id_list = sorted([i["query_id"] for i in matching_list])
                _dataset_id_list = sorted(self.extract_source_identifiers_from_anndata(dataset_path=dataset_path))

                # if the id lists are different than the provided one.
                switch_inconsistency = _saved_id_list != _dataset_id_list

            if switch_inconsistency is not False:
                if switch_inconsistency is None and self.verbose > 2:
                    print(f"Pickle not found: `{dataset_name}`. Calculating IDTrack matchings.")
                elif switch_inconsistency is True and self.verbose > 2:
                    print(f"Inconsistent IDs with pickle object and the dataset: `{dataset_name}`. Calculating IDTrack matchings.")
                matching_list = self.run_idtrack_for_single_dataset(dataset_name, dataset_path)
                with open(dataset_pickle, "wb") as handle:
                    pickle.dump(matching_list, handle)

            result_dataset[dataset_name] = matching_list
            if self.verbose > 1:
                binned_conversions = self.idt.classify_multiple_conversion(matching_list)
                print(f"{dataset_name}")
                self.idt.print_binned_conversion(binned_conversions)
                print()
            
            # Do not allow n_to_1 within a certain dataset.
            self.n_to_1_within_individual_dataset(dataset_name=dataset_name, dataset_matching_list=matching_list)

        return result_dataset

    def extract_source_identifiers_from_anndata(self, dataset_path):
        adata = ad.read_h5ad(dataset_path, backed='r')
        result = list(adata.var_names)
        del adata
        gc.collect()

        return result

    def run_idtrack_for_single_dataset(self, dataset_name, dataset_path) -> List[dict]:
        self._initialize_idt()
        gene_list = self.extract_source_identifiers_from_anndata(dataset_path)

        matching_list = self.idt.convert_identifier_multiple(
            gene_list, final_database=self.final_database, to_release=self.target_ensembl_release, 
            pbar_prefix=dataset_name, verbose=self.verbose > 0
        )

        return matching_list

    def n_to_1_within_individual_dataset(self, dataset_name, dataset_matching_list):
        reverse_dict_target = dict()
        reverse_dict_alternative_target = dict()

        for matching_entry in dataset_matching_list:
            
            # to capture matching without alternative target database.
            # note that final database is `None` for 1-to-0, `ensembl_gene` for alternative target database.
            
            query_id = matching_entry["query_id"]
            
            if matching_entry["final_database"] == self.final_database:
                for i, j in matching_entry["last_node"]:  # for loop if it is 1-to-n
                    if j not in reverse_dict_target:
                        reverse_dict_target[j] = []
                    reverse_dict_target[j].append(query_id)
                    
            elif matching_entry["final_database"] == "ensembl_gene":
                for i, _ in matching_entry["last_node"]:  # for loop if it is 1-to-n
                    
                    if i not in reverse_dict_alternative_target:  
                        reverse_dict_alternative_target[i] = []
                    reverse_dict_alternative_target[i].append(query_id)

        # simply get the ids that has 1-to-n in reverse orientation, so get n-to-1
        
        set_n_to_1_in_dataset = set()
        
        for the_dict_of_interest in [reverse_dict_target, reverse_dict_alternative_target]:
            for matched_id, query_id_list in the_dict_of_interest.items():
                if len(query_id_list) > 1:
                    for query_id in query_id_list:
                        
                        # if there is query and mathed id are the same, exlude this query id from the n-to-1 set. 
                        # basically assign it to be the true matching of the conversion for integration tasks
                        if query_id != matched_id:  # and matching is 1:1. note that if matching is actually 1-to-n, it is resolved by other parts of the code.
                            set_n_to_1_in_dataset.add(query_id)
                    
                    # For debugging and reporting
                    if self.debugging_variables:
                        query_id_list = tuple(sorted(query_id_list))
                        if matched_id in query_id_list:
                            
                            if matched_id not in self.dict_n_to_1_with_query:
                                self.dict_n_to_1_with_query[matched_id] = dict()
                            if query_id_list not in self.dict_n_to_1_with_query[matched_id]:
                                self.dict_n_to_1_with_query[matched_id][query_id_list] = list()    
                            self.dict_n_to_1_with_query[matched_id][query_id_list].append(dataset_name)
                            
                            for query_id in query_id_list:
                                if query_id not in self.dict_n_to_1_with_query_reverse:
                                    self.dict_n_to_1_with_query_reverse[query_id] = dict()
                                if matched_id not in self.dict_n_to_1_with_query_reverse[query_id]:
                                    self.dict_n_to_1_with_query_reverse[query_id][matched_id] = list()    
                                self.dict_n_to_1_with_query_reverse[query_id][matched_id].append(dataset_name)
                                    
                        else:
                            if query_id_list not in self.dict_n_to_1_without_query:
                                self.dict_n_to_1_without_query[query_id_list] = list()    
                            self.dict_n_to_1_without_query[query_id_list].append(dataset_name)

        for nto1 in set_n_to_1_in_dataset:
            if nto1 not in self.dict_n_to_1:
                self.dict_n_to_1[nto1] = []
            self.dict_n_to_1[nto1].append(dataset_name)



def unify_multiple_anndatas(data_h5ad, feature_harmonizer, mode: str, metadata_group_list: list):
    """Unifies the datasets.

    mode='union'. combines different Anndata objects in such a way that the union of all features is preserved
    mode='intersect'. it should intersect the features instead. This means that if a feature is not present in
    one of the studies, it should be removed from the final dataset.

    Args:
        data_h5ad: Todo.
        feature_harmonizer: Todo.
        mode: Todo.

    Returns:
        Todo.
    """
    handle_anndata_key = "handle_anndata"
    _adata_var = pd.DataFrame()
    _adata_obs = pd.DataFrame(columns=[handle_anndata_key] + metadata_group_list)
    _adata_X = csr_matrix(np.zeros((0, 0)).astype(np.float32))
    adata = ad.AnnData(X=_adata_X, obs=_adata_obs, var=_adata_var)

    remove_var_columns = list()
    pbar = tqdm.tqdm(data_h5ad.items())
    for ind, (handle, _) in enumerate(pbar):
        adata_study, t0, t1 = feature_harmonizer.feature_harmonizer(dataset_name = handle)
        pbar.set_postfix(
            {
                "dataset": handle,
                "study_var": adata_study.n_vars,
                "union_var": adata.n_vars,
                "dbh": (t0-t1),  # deleted by harmonizer
            }
        )
        pbar.refresh()
        adata_study.obs["sample_ID"] = adata_study.obs["sample_ID"].astype(str)
        adata_study.obs[handle_anndata_key] = np.full_like(adata_study.obs["sample_ID"], handle)
        adata_study.obs["unified_index"] = f"{handle}_" + adata_study.obs.index.astype(str)
        adata_study.obs.set_index("unified_index", drop=True, inplace=True)

        if "HGNC Symbol" not in adata_study.var.columns:
            raise NotADirectoryError("Final database other than HGNC is not tested.")
        adata_study.var.drop(columns=["Query ID"], inplace=True)
        new_hgnc_var_column = f"{GENE_NAME_COLUMN}_{handle}"
        adata_study.var.rename(columns={"HGNC Symbol": new_hgnc_var_column}, inplace=True, errors="raise")

        if mode == "union" or ind == 0:
            var_map = pd.merge(
                adata.var,
                adata_study.var,
                how="outer",
                left_index=True,
                right_index=True,
            )

            adata = ad.concat(
                adatas=[adata, adata_study],
                join="outer",
                fill_value=0.0,
            )
            
            if mode != "union":
                remove_var_columns.append(new_hgnc_var_column)

        elif mode == "intersect":
            var_map = pd.merge(
                adata.var,
                adata_study.var,
                how="inner",  # changed from "outer" to "inner"
                left_index=True,
                right_index=True,
            )

            adata = ad.concat(
                adatas=[adata, adata_study],
                join="inner",  # changed from "outer" to "inner"
            )
            remove_var_columns.append(new_hgnc_var_column)

        else:
            raise ValueError

        # create a mask to array and put it in `uns`, to specify which are added `0`s because of the `idtrack` process.
        # not needed: look at the adata.var

        assert not np.any(var_map.index.duplicated())
        adata.var = var_map.loc[adata.var_names]

        del adata_study
        gc.collect()

    # Check unified gene naming
    assert len(set(adata.var.index)) == len(adata.var.index)
    gene_name_columns = [i for i in adata.var if i.startswith(f"{GENE_NAME_COLUMN}_")]
    gene_name_columns = adata.var[gene_name_columns].values
    gene_names_unified = list()
    gene_names_inconsistency = dict()
    for ind_i, i in enumerate(gene_name_columns):
        i = i[~pd.isna(i)]
        if len(set(i)) != 1:
            gene_names_inconsistency[adata.var.index[ind_i]] = {
                adata.var.columns[j].split(f"{GENE_NAME_COLUMN}_")[1]: i[j] for j in range(len(i))
            }
        else:
            gene_names_unified.append(i[0])
    assert len(gene_names_inconsistency) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = adata[:, ~adata.var.index.isin(gene_names_inconsistency.keys())]
        assert GENE_NAME_COLUMN not in adata.var.columns
        adata.var[GENE_NAME_COLUMN] = gene_names_unified
        assert GENE_NAME_COLUMN in adata.var.columns
    
    # For `intersect` mode, all these columns are gonna be the same, so delete.
    if len(remove_var_columns) > 0:
        adata.var = adata.var.drop(columns=remove_var_columns)

    # Return
    assert pd.to_numeric(adata.obs["age"], errors='coerce').notna().all(), "Non-numeric value in `age`."
    adata.obs["age"] = adata.obs["age"].astype(float)
    
    for column in adata.obs.columns:
        if column in ["age"]:
            continue    
        # Convert to string, remove categories
        adata.obs[column] = adata.obs[column].astype(str)
        # Replace None or 'nan' values with 'NA'
        adata.obs[column].replace(MISSING_VALUES, NA_CELL_TYPE_PLACEHOLDER, inplace=True)
        # Convert back to categorical
        adata.obs[column] = adata.obs[column].astype('category')
    
    for column in adata.var.columns:
        adata.var[column] = adata.var[column].astype(str)
        adata.var[column].replace(MISSING_VALUES, NA_CELL_TYPE_PLACEHOLDER, inplace=True)
        adata.var[column] = adata.var[column].astype('category')
    
    if mode == "union":
        intersect_column_name = "intersection"
        adata.var[intersect_column_name] = create_intersection_column_values(adata.var.copy())
        adata.var[intersect_column_name] = adata.var[intersect_column_name].astype(int)
    
    return adata

def create_intersection_column_values(adata_var):
    gene_name_columns = [i for i in adata_var.columns if i.startswith(f"{GENE_NAME_COLUMN}_")]
    df = adata_var[gene_name_columns].copy()
    df = df != NA_CELL_TYPE_PLACEHOLDER
    return (df.sum(axis=1) == len(gene_name_columns)).astype(int).values
