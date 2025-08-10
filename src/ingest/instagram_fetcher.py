import os
import time
import random
import pandas as pd
from instagrapi import Client

from enums.enum import Directories, Const
from ..common.versioning import dated_subdir
from ..common.text_norm import normalize_fa


class InstagramFetcher:
    def __init__(
        self,
        logger,
        download_media: bool = True,
        amount: int = Const.INSTAGRAM_POST_COUNT.value,   # تعداد پست‌ها برای full refresh
        delay_range: tuple[float, float] = (2, 5),
    ):
        self.log = logger
        self.download_media = download_media
        self.amount = int(amount)
        self.delay_range = delay_range

        # env creds (در main از dotenv لود می‌شود)
        self.username = os.getenv("INSTAGRAM_USERNAME")
        self.password = os.getenv("INSTAGRAM_PASSWORD")
        if not self.username or not self.password:
            raise RuntimeError("ENV: INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD لازم است.")

        # مسیرها بر اساس Enum
        self.raw_root = Directories.RAW_INSTAGRAM.value
        self.session_dir = Directories.SESSIONS.value
        os.makedirs(self.session_dir, exist_ok=True)
        self.session_file = os.path.join(self.session_dir, "settings.json")

    # ---------- helpers ----------
    def _extract_insight_metrics(self, insight: dict) -> dict:
        metrics = insight.get("inline_insights_node", {}).get("metrics", {})

        tray_nodes = metrics.get("share_count", {}).get("tray", {}).get("nodes", []) or []
        tray_shares = sum(node.get("value", 0) for node in tray_nodes)
        post_shares = metrics.get("share_count", {}).get("post", {}).get("value", 0) or 0
        total_shares = tray_shares + post_shares

        return {
            "instagram_media_type": insight.get("instagram_media_type", "UNKNOWN"),
            "save_count": insight.get("save_count", 0),
            "impressions_count": metrics.get("impression_count", 0),
            "reach_count": metrics.get("reach", {}).get("value", 0),
            "profile_views": metrics.get("owner_profile_views_count", 0),
            "follows_from_post": metrics.get("owner_account_follows_count", 0),
            "impressions_feed": next((n.get("value", 0) for n in metrics.get("impressions", {}).get("surfaces", {}).get("nodes", []) if n.get("name") == "FEED"), 0),
            "impressions_profile": next((n.get("value", 0) for n in metrics.get("impressions", {}).get("surfaces", {}).get("nodes", []) if n.get("name") == "PROFILE"), 0),
            "impressions_explore": next((n.get("value", 0) for n in metrics.get("impressions", {}).get("surfaces", {}).get("nodes", []) if n.get("name") == "EXPLORE"), 0),
            "bio_link_clicks": next((n.get("value", 0) for n in metrics.get("profile_actions", {}).get("actions", {}).get("nodes", []) if n.get("name") == "BIO_LINK_CLICKED"), 0),
            "share_count": total_shares,
        }

    def _login(self) -> Client:
        self.log.info("🔐 Instagram: login...")
        cl = Client()
        cl.set_locale("en_US")
        cl.set_country("US")

        if os.path.exists(self.session_file):
            try:
                cl.load_settings(self.session_file)
                cl.login(self.username, self.password)
                # cl.inject_sessionid_to_public()
                self.log.info("✅ Logged in using session file.")
            except Exception as e:
                self.log.warning(f"Session login failed: {e}. Trying fresh login...")
                cl.login(self.username, self.password)
                # cl.inject_sessionid_to_public()
                cl.dump_settings(self.session_file)
                self.log.info("✅ Fresh login & session saved.")
        else:
            cl.login(self.username, self.password)
            # cl.inject_sessionid_to_public()
            cl.dump_settings(self.session_file)
            self.log.info("✅ Logged in and session saved.")

        cl.delay_range = list(self.delay_range)
        return cl

    # ---------- main ----------
    def full_refresh(self) -> pd.DataFrame:
        cl = self._login()


        user_id = cl.user_id_from_username(self.username)
        self.log.info(f"📥 Fetching last {self.amount} medias for @{self.username} (user_id={user_id}) ...")
        medias = cl.user_medias(user_id, amount=self.amount)
        # medias = cl.user_medias_gql(user_id, amount=50)
        self.log.info(f"✅ Retrieved {len(medias)} medias.")

        dated_dir = dated_subdir(self.raw_root)  # e.g., data/raw/instagram/20250809
        out_csv = os.path.join(dated_dir, "instagram_posts.csv")

        photos_dir = os.path.join(dated_dir, "downloads", "photos")
        videos_dir = os.path.join(dated_dir, "downloads", "videos")
        carousel_dir = os.path.join(dated_dir, "downloads", "carousel")
        if self.download_media:
            os.makedirs(photos_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(carousel_dir, exist_ok=True)

        rows = []
        for idx, media in enumerate(medias, start=1):
            self.log.info(f"📄 Processing {idx}/{len(medias)} (ID: {media.id})")
            time.sleep(random.uniform(*self.delay_range))

            # Insights
            try:
                media_insight = cl.insights_media(media.pk)
            except Exception as e:
                self.log.warning(f"insights_media failed for {media.id}: {e}")
                media_insight = {}
            insight_data = self._extract_insight_metrics(media_insight)

            file_path = ""
            if self.download_media:
                try:
                    if media.media_type == 1:
                        file_path = cl.photo_download(media.id, folder=photos_dir)
                    elif media.media_type == 2:
                        file_path = cl.video_download(media.id, folder=videos_dir)
                    elif media.media_type == 8:
                        file_path = cl.album_download(media.id, folder=carousel_dir)
                    if file_path:
                        self.log.info(f"📥 Downloaded → {file_path}")
                except Exception as e:
                    self.log.warning(f"Download failed for {media.id}: {e}")

            rows.append({
                "id": str(media.id),
                "code": media.code,
                "url": f"https://www.instagram.com/p/{media.code}/",
                "taken_at": media.taken_at.strftime("%Y-%m-%d %H:%M:%S"),
                "username": media.user.username,
                "media_type_id": media.media_type,
                "caption": media.caption_text,
                "like_count": media.like_count,
                "comment_count": media.comment_count,
                "play_count": media.play_count if media.media_type == 2 else None,
                "video_duration": media.video_duration if media.media_type == 2 else None,
                "image_url": media.thumbnail_url,
                "downloaded_file": file_path or None,
                **insight_data
            })

        df = pd.DataFrame(rows)
        # df["username"] = df["username"].map(normalize_fa)

        # ذخیره CSV تاریخ‌دار
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        self.log.info(f"💾 Saved raw instagram CSV → {out_csv} ({len(df)} rows)")

        return df
