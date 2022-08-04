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
    mysql_port = 3306
    mysql_port_min_version = 48  # 47 and below is 'mysql_port_before_rel'.

    # Protected Non-int Version Strings
    no_old_node_id = "Void"
    no_new_node_id = "Retired"
    alternative_versions = {no_new_node_id, no_old_node_id}

    external_search_settings = {"jump_limit": 2, "synonymous_max_depth": 3, "backbone_node_type": "ensembl_gene"}
