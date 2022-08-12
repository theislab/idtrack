#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


class DB:
    """Keeps some variables shared across different classes."""

    # Delimiter separating the ID and the version of an ensembl identifier.
    id_ver_delimiter = "."
    first_version: int = 1  # just and arbitrary assumption

    # FTP server from which the above databases will be downloaded.
    ensembl_ftp_base = "ftp.ensembl.org"

    # Ensembl REST API to fetch information of organisms and latest releases.
    rest_server_api = "https://rest.ensembl.org"
    rest_server_ext = "/info/species?"

    # MYSQL
    mysql_host = "ensembldb.ensembl.org"
    myqsl_user = "anonymous"
    mysql_togo = ""
    mysql_port_and_assembly_priority = {  # assembly -> [mysql_port, assembly priority]
        38: [3306, 1, 48],  # From Ensembl 48 onwards only
        37: [3337, 2, 79],  # Databases for archive GRCh37 - release 79 onwards
    }

    # Protected Non-int Version Strings
    no_old_node_id = "Void"
    no_new_node_id = "Retired"
    alternative_versions = {no_new_node_id, no_old_node_id}

    # Node Types
    node_type_str = "node_type"
    nts_external = "external"
    nts_assembly = {
        j: {i: f"assembly_{j}_ensembl_{i}" for i in ["gene", "transcript", "translation"]}
        for j in mysql_port_and_assembly_priority
    }
    nts_ensembl = {i: f"ensembl_{i}" for i in ["gene", "transcript", "translation"]}
    nts_base_ensembl = {i: f"base_ensembl_{i}" for i in ["gene", "transcript", "translation"]}

    # Edge/Note Attributes:
    # Todo:

    external_search_settings = {"jump_limit": 2, "synonymous_max_depth": 3, "backbone_node_type": "ensembl_gene"}
