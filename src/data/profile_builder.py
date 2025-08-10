import numpy as np, pandas as pd
from enums.enum import Directories
from ..common.text_norm import normalize_fa
from ..data.io import read_parquet, write_parquet

def _safe_div(a,b): a=a.fillna(0); b=b.replace(0,np.nan); return (a/b).fillna(0)

def build_product_profile() -> pd.DataFrame:
    latest_instagram = Directories.STAGING_INSTAGRAM.value
    sales_merged     = Directories.STAGING_SALES.value
    map_path         = Directories.RAW_PRODUCT_MAP.value
    out_profile      = Directories.MART_PRODUCT_PROFILE.value

    media = read_parquet(latest_instagram)
    sales = read_parquet(sales_merged)
    pmap  = pd.read_excel(map_path)

    sales = sales.rename(columns={"product_name":"product_name_orig"})
    sales["product_key"] = sales["product_name_orig"].map(normalize_fa)

    pmap  = pmap.rename(columns={"product":"product_name_orig"})
    pmap["product_key"] = pmap["product_name_orig"].map(normalize_fa)
    pmap["media_id"] = pmap["media_id"].astype(str).str.strip()

    media = media.rename(columns={"id":"media_id"})
    media["media_id"] = media["media_id"].astype(str).str.strip()

    metrics = ["like_count","comment_count","play_count","impressions_count","reach_count","save_count","share_count"]
    for c in metrics:
        if c not in media.columns: media[c]=np.nan
        media[c] = pd.to_numeric(media[c], errors="coerce")
    if "media_type" not in media.columns and "media_type_id" in media.columns:
        media["media_type"] = media["media_type_id"].astype(str)
    media = media.sort_values("media_id").drop_duplicates(subset=["media_id"], keep="last")

    sales_agg = (sales.groupby("product_key", as_index=False)
                 .agg(total_sales=("quantity","sum"), sale_rows=("quantity","count")))

    pm = pmap.merge(media, on="media_id", how="left")

    posts_by_type = (pm.groupby(["product_key","media_type"], dropna=False).size().reset_index(name="count")
                     .pivot_table(index="product_key", columns="media_type", values="count",
                                  aggfunc="sum", fill_value=0).rename_axis(None, axis=1).reset_index())

    agg = (pm.groupby("product_key", as_index=False)
             .agg(total_posts=("media_id","nunique"),
                  like_count=("like_count","sum"),
                  comment_count=("comment_count","sum"),
                  save_count=("save_count","sum"),
                  share_count=("share_count","sum"),
                  impressions_count=("impressions_count","sum"),
                  reach_count=("reach_count","sum"),
                  play_count=("play_count","sum")))

    agg["avg_likes"]      = _safe_div(agg["like_count"], agg["total_posts"])
    agg["avg_comments"]   = _safe_div(agg["comment_count"], agg["total_posts"])
    agg["avg_saves"]      = _safe_div(agg["save_count"], agg["total_posts"])
    agg["avg_shares"]     = _safe_div(agg["share_count"], agg["total_posts"])
    agg["avg_impr_post"]  = _safe_div(agg["impressions_count"], agg["total_posts"])
    agg["avg_reach_post"] = _safe_div(agg["reach_count"], agg["total_posts"])
    agg["avg_plays_post"] = _safe_div(agg["play_count"], agg["total_posts"])
    denom = agg["impressions_count"].replace(0,np.nan)
    agg["engagement_rate"] = ((agg["like_count"]+agg["comment_count"]+agg["save_count"]+agg["share_count"]) / denom).fillna(0)

    profile = (agg.merge(posts_by_type, how="left", on="product_key")
                  .merge(sales_agg,    how="left", on="product_key"))

    for c in profile.columns:
        if c!="product_key" and pd.api.types.is_numeric_dtype(profile[c]): profile[c]=profile[c].fillna(0)

    write_parquet(profile, out_profile)
    return profile
