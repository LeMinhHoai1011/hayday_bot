"""Shared, evidence-driven farm/shop workflows. Missing evidence always stops input."""
import subprocess
import hashlib
import json
import re
import time
from pathlib import Path
import cv2
import numpy as np
from bot_config import CONFIG, resolve

class Stopped(RuntimeError):
    pass

class EvidenceError(RuntimeError):
    pass


class PlotActionDeferred(EvidenceError):
    pass

HARVEST_TEMPLATES = ('farm_screen', 'field_ripe', 'harvest_tool', 'field_empty', 'field_selected_ripe')
PLANT_TEMPLATES = ('farm_screen', 'field_empty', 'seed_menu', 'crop_menu_wheat', 'field_growing',
                   'field_selected_empty')
SHOP_TEMPLATES = ('farm_screen', 'shop_stand', 'shop_screen', 'shop_slot_empty',
                  'shop_close', 'inventory_screen', 'inventory_wheat', 'inventory_close',
                  'wheat_selected', 'btn_price_max', 'price_max_reached',
                  'btn_put_sale', 'shop_slot_wheat')
WORKFLOW_ERRORS = (RuntimeError, OSError, subprocess.SubprocessError, cv2.error)


def intersection(first, second):
    return (max(first[0], second[0]), max(first[1], second[1]),
            min(first[2], second[2]), min(first[3], second[3]))


def near(point, target):
    tolerance = CONFIG.get('field_center_tolerance', 8)
    return max(abs(a-b) for a, b in zip(point, target)) <= tolerance


class Vision:
    def __init__(self, bot):
        self.bot = bot
        self.templates = {}

    def template(self, name):
        if name not in self.templates:
            path = resolve(CONFIG['templates_dir']) / (name + '.png')
            image = cv2.imread(str(path)) if path.is_file() else None
            if image is None or min(image.shape[:2]) < 2 or not np.any(image != image[0, 0]):
                return None
            self.templates[name] = image
        return self.templates[name]

    def require(self, names):
        missing = [name + '.png' for name in names if self.template(name) is None]
        if missing:
            raise EvidenceError('missing_template: ' + ', '.join(missing)
                                + ' | Open the matching state in BlueStacks and crop it using python run.py, menu 2.')

    def bounds(self, name, image, roi=None):
        h, w = image.shape[:2]
        configured = CONFIG['rois'].get(name, (0, 0, 1, 1))
        if not isinstance(configured, (list, tuple)) or len(configured) != 4 or not (0 <= configured[0] < configured[2] <= 1
                                       and 0 <= configured[1] < configured[3] <= 1):
            raise EvidenceError('invalid_roi:' + name)
        bounds = configured if roi is None else intersection(configured, roi)
        return tuple(round(v * n) for v, n in zip(bounds, (w, h, w, h)))

    def matches(self, name, image, roi=None):
        self.require((name,))
        template = self.template(name)
        x0, y0, x1, y1 = self.bounds(name, image, roi)
        if x0 >= x1 or y0 >= y1:
            return []
        crop = image[y0:y1, x0:x1]
        th, tw = template.shape[:2]
        if crop.shape[0] < th or crop.shape[1] < tw:
            return []
        threshold = CONFIG.get('match_thresholds', {}).get(name, CONFIG['match_threshold'])
        comparison, pattern = crop, template
        if name in ('field_ripe', 'field_growing'):
            comparison = cv2.GaussianBlur(crop, (9, 9), 0)
            pattern = cv2.GaussianBlur(template, (9, 9), 0)
        if name == 'harvest_tool':
            # Match the silver blade silhouette across grass, soil and shop-roof backgrounds.
            comparison = self.silver_mask(comparison)
            pattern = self.silver_mask(pattern)
        if name.startswith('field_selected_'):
            comparison = cv2.GaussianBlur(
                cv2.inRange(comparison, (210, 210, 235), (255, 255, 255)), (9, 9), 0)
            pattern = cv2.GaussianBlur(
                cv2.inRange(pattern, (210, 210, 235), (255, 255, 255)), (9, 9), 0)
        scores = cv2.matchTemplate(comparison, pattern, cv2.TM_CCOEFF_NORMED)
        found = []
        while len(found) < 100:
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if not np.isfinite(score) or score < threshold:
                break
            patch = crop[y:y+th, x:x+tw]
            if self.accept_match(name, patch, template):
                found.append((x0+x+tw//2, y0+y+th//2, float(score)))
            scores[max(0,y-th//2):y+th//2+1, max(0,x-tw//2):x+tw//2+1] = -1
        return found

    @staticmethod
    def silver_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.GaussianBlur(cv2.inRange(hsv, (0, 0, 155), (179, 65, 255)), (7, 7), 0)

    @staticmethod
    def accept_match(name, patch, template):
        if name in ('farm_screen', 'btn_put_sale', 'btn_price_max', 'btn_quantity_minus',
                    'price_max_reached_any_quantity'):
            return cv2.absdiff(patch, template).mean() <= 24
        if name == 'field_empty':
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            soil = cv2.inRange(hsv, (0, 80, 35), (23, 255, 245))
            return cv2.countNonZero(soil) >= soil.size * .75
        if name == 'field_ripe':
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            wheat = cv2.inRange(hsv, (17, 95, 170), (42, 255, 255))
            return cv2.countNonZero(wheat) >= wheat.size * .4
        return True

    def point(self, name, image, roi=None):
        matches = self.matches(name, image, roi)
        if not matches:
            raise EvidenceError('unconfirmed:' + name)
        return matches[0][:2]

    def drag_point(self, name, center):
        anchor = CONFIG.get('drag_anchors', {}).get(name)
        if anchor is None:
            return center
        if len(anchor) != 2 or not all(0 <= value <= 1 for value in anchor):
            raise EvidenceError('invalid_drag_anchor:' + name)
        h, w = self.template(name).shape[:2]
        x, y = center
        return (x-w//2+round(anchor[0]*(w-1)), y-h//2+round(anchor[1]*(h-1)))

    def state(self, name, image):
        return bool(self.matches(name, image))

    def tool_open(self, image):
        if self.template('harvest_tool') is not None and self.state('harvest_tool', image):
            return True
        return all(self.template(name) is not None and self.state(name, image)
                   for name in ('seed_menu', 'crop_menu_wheat'))

    def camera(self, before, after, excluded=()):
        if before.shape != after.shape:
            raise EvidenceError('screen_size_changed')
        source = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        target = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        h, w = source.shape
        mask = np.zeros_like(source)
        # Fixed HUD icons can outvote real map features during the game's large pans.
        mask[int(h*.15):int(h*.82), int(w*.12):int(w*.89)] = 255
        for x0, y0, x1, y1 in excluded:
            mask[max(0, y0-24):y1+24, max(0, x0-24):x1+24] = 0
        points = cv2.goodFeaturesToTrack(source, 32, .02, max(30, w//22), mask=mask)
        if points is None or len(points) < 8:
            raise EvidenceError('missing_camera_landmarks')

        radius = max(12, w//80)
        reach_x, reach_y = round(w*.22), round(h*.36)
        old, new = [], []
        for x, y in np.rint(points[:, 0]).astype(int):
            if not (radius <= x < w-radius and radius <= y < h-radius):
                continue
            patch = source[y-radius:y+radius+1, x-radius:x+radius+1]
            left, top = max(0, x-reach_x-radius), max(0, y-reach_y-radius)
            right, bottom = min(w, x+reach_x+radius+1), min(h, y+reach_y+radius+1)
            scores = cv2.matchTemplate(target[top:bottom, left:right], patch,
                                       cv2.TM_CCOEFF_NORMED)
            _, score, _, (mx, my) = cv2.minMaxLoc(scores)
            if score < .87:
                continue
            # Reject ambiguous repeat textures at distinct locations.
            scores[max(0,my-5):my+6, max(0,mx-5):mx+6] = -1
            if score - float(scores.max()) < .035:
                continue
            old.append((x, y))
            new.append((mx+left+radius, my+top+radius))
        if len(old) < 8:
            raise EvidenceError('lost_camera_landmarks')
        old, new = np.float32(old), np.float32(new)
        transform, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=2)
        if transform is None or int(inliers.sum()) < max(8, len(old)*.6):
            raise EvidenceError('inconsistent_camera_landmarks')
        if abs(transform[0,0]-1) > .01 or abs(transform[0,1]) > .01:
            raise EvidenceError('camera_zoom_or_rotation')
        dx, dy = np.median((new-old)[inliers.ravel() == 1], axis=0)
        return float(dx), float(dy)


class Workflow:
    def __init__(self, bot):
        self.b = bot
        self.v = Vision(bot)
        self.reason = 'ready'
        self.last_frame = None
        self.previous_frame = None
        self.action_frame = None
        self.debug = resolve(CONFIG['debug_dir'])
        self.debug.mkdir(parents=True, exist_ok=True)

    def log(self, action, **details):
        record = dict(time=time.time(), action=action, device=self.b.device, **details)
        with (self.debug/'steps.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(record, ensure_ascii=False))

    def check(self):
        self.b.check_stop()

    def stable(self):
        self.check()
        previous = self.capture()
        for _ in range(16):
            self.check()
            time.sleep(.2)
            current = self.capture()
            if previous.shape == current.shape:
                # Ignore local effects while still waiting for movement across the farm view.
                difference = cv2.absdiff(previous, current).astype(np.float32).mean(axis=2)
                blocks = cv2.resize(difference, (12, 8), interpolation=cv2.INTER_AREA)
                if float(np.median(blocks)) <= CONFIG['stable_threshold']:
                    return current
            previous = current
        raise EvidenceError('screen_not_stable')

    def capture(self):
        self.previous_frame = self.last_frame
        self.last_frame = self.b.screenshot()
        return self.last_frame

    def tap(self, point, source):
        self.check()
        self.log('tap', coordinates=point, source=source, before=source)
        self.action_frame = self.last_frame
        self.b.tap(*point)

    def fail(self, error, before=None):
        self.reason = str(error)
        self.b.last_reason = self.reason
        stamp = str(time.time_ns())
        before = before if before is not None else (self.action_frame if self.action_frame is not None else self.previous_frame)
        after = self.last_frame
        if before is not None:
            cv2.imwrite(str(self.debug/(stamp+'_before.png')), before)
        if after is not None:
            cv2.imwrite(str(self.debug/(stamp+'_after.png')), after)
        self.log('stop', reason=self.reason)

class FarmWorkflow(Workflow):
    def __init__(self, bot, zones=None, reference=None):
        super().__init__(bot)
        self.zones = zones
        self.reference = reference
        self.manual_zones = zones is not None

    @staticmethod
    def fraction(box, image):
        h, w = image.shape[:2]
        return tuple(v/n for v, n in zip(box, (w, h, w, h)))

    @property
    def calibration_path(self):
        serial = hashlib.sha256(self.b.device.encode()).hexdigest()[:12]
        return self.debug / f'field_calibration_{serial}.npz'

    def save_calibration(self, image, bounds):
        path = self.calibration_path
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as file:
            np.savez_compressed(file, frame=image, bounds=bounds)
        temporary.replace(path)

    def regions(self, image):
        if self.zones is not None:
            return self.zones
        try:
            with np.load(self.calibration_path) as calibration:
                self.reference = calibration['frame']
                box = tuple(map(int, calibration['bounds']))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise EvidenceError('field_region_required: calibrate with auto_hayday.py menu 9 or select a GUI zone') from error
        if self.reference.shape != image.shape:
            self.reselect()
        self.zones = [box]
        return self.zones

    def camera(self, before, after, target=None):
        # Keep the field and transient tool menu out of the camera landmarks.
        excluded = list(self.regions(before))
        if target:
            x, y = target
            excluded.append((x-135, y-110, x+135, y+110))
        return self.v.camera(before, after, excluded)

    def moved(self, dx, dy):
        return max(abs(dx), abs(dy)) > CONFIG['camera_threshold']

    def reselect(self):
        if self.manual_zones:
            raise EvidenceError('camera_moved_reselect_zones: select the field again on a fresh GUI frame')
        raise EvidenceError('camera_moved_recalibrate_field: calibrate with auto_hayday.py menu 9')

    def verify_frame(self, image):
        self.v.point('farm_screen', image)
        if self.v.tool_open(image):
            raise EvidenceError('tool_menu_already_open: close it in BlueStacks and scan again')
        self.regions(image)
        if self.reference is not None and self.reference is not image:
            if self.reference.shape != image.shape:
                self.reselect()
            try:
                displacement = self.camera(self.reference, image)
            except EvidenceError as error:
                if self.manual_zones:
                    self.reselect()
                raise error
            if self.moved(*displacement):
                self.reselect()

    def fields(self, name, image):
        matches = {}
        for box in self.regions(image):
            for x, y, score in self.v.matches(name, image, self.fraction(box, image)):
                matches[x, y] = max(score, matches.get((x, y), 0))
        return sorted(((x, y, score) for (x, y), score in matches.items()),
                      key=lambda point: (point[1], point[0]))

    def fields_at(self, name, image, targets):
        points = []
        for x, y in targets:
            local = (x-37, y-33, x+37, y+33)
            found = []
            for field in self.regions(image):
                bounds = self.fraction(intersection(field, local), image)
                found.extend(p for p in self.v.matches(name, image, bounds) if near(p[:2], (x, y)))
            if found:
                points.append(max(found, key=lambda p: p[2]))
        return sorted(points, key=lambda p: (p[1], p[0]))

    def preview(self, image, name='field_ripe'):
        self.v.require(('farm_screen', name))
        self.verify_frame(image)
        points = self.fields(name, image)
        overlay = image.copy()
        for box in self.regions(image):
            x0, y0, x1, y1 = self.v.bounds(name, image, self.fraction(box, image))
            if x0 < x1 and y0 < y1:
                cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), (255, 255, 0), 2)
        for x, y, _ in points:
            cv2.drawMarker(overlay, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        if points:
            x, y = points[0][:2]
            cv2.circle(overlay, (x, y), 10, (0, 0, 255), 2)
        return overlay, points

    def shift_zones(self, dx, dy, image):
        if self.manual_zones:
            self.reselect()
        h, w = image.shape[:2]
        shifted = [tuple(round(v+d) for v, d in zip(box, (dx, dy, dx, dy)))
                   for box in self.zones]
        if any(not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) for x0,y0,x1,y1 in shifted):
            self.reselect()
        self.zones = shifted
        self.reference = image
        self.log('redetect', reason='camera_moved', displacement=[dx, dy])

    def confirm_transition(self, menu, point, previous, result):
        x, y = point
        # Harvested bundles and the planting highlight can temporarily cover the same plot.
        for _ in range(4):
            after = self.stable()
            self.v.point('farm_screen', after)
            dx, dy = self.camera(menu, after, point)
            target = (x+dx, y+dy)
            tx, ty = target
            local = self.fraction((tx-37, ty-33, tx+37, ty+33), after)
            confirmed = any(near(p[:2], target) for p in self.v.matches(result, after, local))
            still_previous = any(near(p[:2], target) for p in
                                 self.v.matches(previous, after, local))
            if confirmed and not still_previous:
                return after, dx, dy, target
            self.check()
        raise EvidenceError('target_state_unconfirmed:' + result)

    def run(self, plant=False, zones=None, limit=None, targets=None):
        count = 0
        self.confirmed = []
        targets = list(targets) if targets is not None else None
        self.action_frame = None
        self.reason = 'ready'
        before = None
        if zones is not None:
            self.zones = zones
            self.manual_zones = True
        state = 'field_empty' if plant else 'field_ripe'
        result = 'field_growing' if plant else 'field_empty'
        try:
            self.v.require(PLANT_TEMPLATES if plant else HARVEST_TEMPLATES)
            if limit is not None and limit < 1:
                raise EvidenceError('field limit must be positive')
            with self.b.task():
                for _ in range(300):
                    before = self.stable()
                    self.verify_frame(before)
                    candidates = (self.fields(state, before) if targets is None
                                  else self.fields_at(state, before, targets))
                    if not candidates and targets:
                        for _ in range(2):
                            before = self.stable()
                            self.verify_frame(before)
                            candidates = self.fields_at(state, before, targets)
                            if candidates:
                                break
                        else:
                            raise EvidenceError('required_target_unconfirmed:' + state)
                    if not candidates:
                        self.reason = 'complete' if count else 'no_confirmed_fields'
                        self.log('farm_result', reason=self.reason, confirmed=count)
                        break
                    x, y = candidates[0][:2]
                    self.tap((x, y), state)
                    menu = self.stable()
                    if plant:
                        self.v.point('seed_menu', menu)
                    tool_name = 'crop_menu_wheat' if plant else 'harvest_tool'
                    tool_center = self.v.point(tool_name, menu)
                    tool = self.v.drag_point(tool_name, tool_center)
                    dx, dy = self.camera(before, menu, (x, y))
                    if self.moved(dx, dy):
                        self.shift_zones(dx, dy, menu)
                        predicted = (x+dx, y+dy)
                        matches = self.fields_at(state, menu, (predicted,))
                        if not matches:
                            selected = 'field_selected_empty' if plant else 'field_selected_ripe'
                            px, py = predicted
                            local = self.fraction((px-65, py-44, px+65, py+44), menu)
                            matches = [p for p in self.v.matches(selected, menu, local)
                                       if near(p[:2], predicted)]
                        if len(matches) != 1:
                            raise EvidenceError('target_lost_after_camera_move')
                        x, y = matches[0][:2]
                        if targets is not None:
                            targets[:] = [(px+dx, py+dy) for px, py in targets]
                        self.confirmed = [(px+dx, py+dy) for px, py in self.confirmed]
                    action = 'plant' if plant else 'harvest'
                    self.check()
                    self.log(action+'_drag', coordinates=[tool,(x,y)], match_center=tool_center, source=tool_name+'/'+state)
                    if self.last_frame is not None:
                        overlay = menu.copy()
                        for box in self.zones:
                            x0, y0, x1, y1 = self.v.bounds(state, menu, self.fraction(box, menu))
                            cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), (255, 255, 0), 2)
                        cv2.circle(overlay, tool_center, 8, (255, 128, 0), 2)
                        cv2.drawMarker(overlay, tool, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                        cv2.putText(overlay, 'DRAG START', (tool[0]+10, tool[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
                        cv2.arrowedLine(overlay, tool, (x, y), (0, 0, 255), 3)
                        cv2.drawMarker(overlay, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
                        cv2.imwrite(str(self.debug/'last_drag.png'), overlay)
                    self.action_frame = menu
                    self.b.drag(*tool, x, y)
                    after, ax, ay, target = self.confirm_transition(menu, (x, y), state, result)
                    self.log('camera', displacement=[ax, ay])
                    count += 1
                    if targets is not None:
                        targets = [p for p in targets if not near(p, (x, y))]
                    self.confirmed.append(target)
                    self.log(action, coordinates=target, source='fresh_template',
                             before=state, after=result, confirmed=count)
                    if self.moved(ax, ay):
                        self.shift_zones(ax, ay, after)
                        self.confirmed[:-1] = [(px+ax, py+ay) for px, py in self.confirmed[:-1]]
                        if targets is not None:
                            targets[:] = [(px+ax, py+ay) for px, py in targets]
                    if not self.manual_zones and (self.moved(dx, dy) or self.moved(ax, ay)):
                        self.reference = after
                        if self.last_frame is not None:
                            self.save_calibration(after, self.zones[0])
                    if limit is not None and count >= limit:
                        self.reason = 'complete'
                        break
                else:
                    raise EvidenceError('operation_limit')
        except WORKFLOW_ERRORS as error:
            self.fail(error, self.action_frame if self.action_frame is not None else before)
        self.b.last_reason = self.reason
        return count


class CalibratedFarm(FarmWorkflow):
    """Track plot IDs in a reference image throughout a harvest and replant pass."""

    @property
    def calibration_path(self):
        return super().calibration_path.with_name(
            super().calibration_path.name.replace('field_calibration_', 'plot_calibration_'))

    def save_plots(self, image, centers):
        h, w = image.shape[:2]
        if (not centers or any(not (16 <= x < w-16 and 16 <= y < h-16) for x, y in centers)
                or any(near(point, other) for i, point in enumerate(centers)
                       for other in centers[:i])):
            raise EvidenceError('invalid_plot_centers')
        self.v.point('farm_screen', image)
        if self.v.tool_open(image):
            raise EvidenceError('tool_menu_already_open')
        path = self.calibration_path
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as file:
            np.savez_compressed(file, frame=image, centers=centers)
        temporary.replace(path)

    def load_plots(self):
        try:
            with np.load(self.calibration_path) as data:
                self.reference = data['frame']
                self.centers = [tuple(map(int, p)) for p in data['centers']]
        except (OSError, ValueError, KeyError) as error:
            raise EvidenceError('plot_calibration_required: auto_hayday.py calibrate-plots') from error
        xs, ys = zip(*self.centers)
        h, w = self.reference.shape[:2]
        self.zones = [(max(0,min(xs)-48), max(0,min(ys)-65),
                       min(w,max(xs)+48), min(h,max(ys)+35))]

    @staticmethod
    def plot_state(image, point):
        x, y = point
        if x < 15 or y < 14 or x+16 > image.shape[1] or y+10 > image.shape[0]:
            return 'unknown'
        patch = image[y-14:y+10, x-15:x+16]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # The lower core samples the plot surface, below the tops of neighboring stalks.
        core = hsv[5:20, 5:26]
        bright = cv2.inRange(hsv, (17, 105, 150), (33, 255, 255))
        soil = cv2.inRange(core, (0, 85, 40), (18, 255, 245))
        green = cv2.inRange(core, (34, 95, 75), (92, 255, 250))
        wheat = cv2.countNonZero(bright) / bright.size
        core_wheat = cv2.countNonZero(bright[5:20, 5:26]) / soil.size
        earth = cv2.countNonZero(soil) / soil.size
        sprouts = cv2.countNonZero(green) / green.size
        if wheat >= .76 and earth <= .17:
            return 'ripe'
        if core_wheat <= .36 and earth >= .67 and sprouts < .12:
            return 'empty'
        if wheat < .42 and sprouts >= .15 and earth >= .10:
            return 'growing'
        # Taller shoots cover the soil, but their blade texture differs from plain grass.
        if wheat < .42 and sprouts >= .78:
            gray = cv2.cvtColor(patch[5:20, 5:26], cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_32F).std() >= 27:
                return 'growing'
        return 'unknown'

    def scan(self, image, *, allow_tool=False):
        if self.reference is None:
            self.load_plots()
        self.v.point('farm_screen', image)
        if not allow_tool and self.v.tool_open(image):
            raise EvidenceError('tool_menu_already_open: close the tool in BlueStacks')
        dx, dy = self.camera(self.reference, image)
        cells = [(round(x+dx), round(y+dy)) for x,y in self.centers]
        return [(point, self.plot_state(image, point)) for point in cells]

    def preview(self, image, name='field_ripe'):
        scan = self.scan(image)
        overlay = image.copy()
        colors = {'ripe': (0,255,255), 'empty': (40,140,255), 'growing': (30,230,30),
                  'unknown': (220,220,220)}
        points = []
        for index, (point, state) in enumerate(scan, 1):
            color = colors[state]
            cv2.drawMarker(overlay, point, color, cv2.MARKER_CROSS, 14, 2)
            cv2.putText(overlay, f'{index} {state}', (point[0]+5,point[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1)
            if state == 'ripe':
                points.append((*point, 1.0))
        return overlay, points

    def draw_drag(self, image, center, start, point, *, planting):
        overlay = image.copy()
        cv2.circle(overlay, center, 8, (255,128,0), 2)
        cv2.drawMarker(overlay, start, (0,0,255), cv2.MARKER_CROSS, 18, 2)
        cv2.arrowedLine(overlay, start, point, (0,0,255), 2)
        cv2.drawMarker(overlay, point, (0,255,255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(overlay, 'DRAG START', (start[0]+8,start[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
        cv2.putText(overlay, 'TARGET', (point[0]+8,point[1]+18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 2)
        cv2.imwrite(str(self.debug/'last_drag.png'), overlay)
        if not planting:
            cv2.imwrite(str(self.debug/'last_harvest_drag.png'), overlay)

    @staticmethod
    def crosses_plot(start, end, point):
        samples = np.linspace(start, end, 45)
        distances = np.abs(samples-np.asarray(point)) / (36, 17)
        return bool(np.any(distances.sum(axis=1) <= 1))

    def open_tool(self, index, cells, planting):
        point = cells[index][0]
        self.tap(point, 'calibrated_plot')
        menu = self.stable()
        self.action_frame = menu
        mapped = self.scan(menu, allow_tool=True)
        tool = 'crop_menu_wheat' if planting else 'harvest_tool'
        if planting:
            self.v.point('seed_menu', menu)
        target = mapped[index][0]
        x, y = target
        h, w = menu.shape[:2]
        if not (15 <= x < w-15 and 15 <= y < h-15):
            raise EvidenceError('plot_left_viewport')
        local = self.fraction((x-360, y-330, x+360, y+330), menu)
        # Tool position varies with the radial menu and edge of the viewport.
        local = intersection(local, (0,0,1,1))
        try:
            center = self.v.point(tool, menu, local)
        except EvidenceError as error:
            if (not planting and str(error) == 'unconfirmed:harvest_tool'
                    and mapped[index][1] == 'ripe' and not self.v.tool_open(menu)):
                raise PlotActionDeferred('tool_unconfirmed_after_tap') from error
            raise
        start = self.v.drag_point(tool, center)
        self.draw_drag(menu, center, start, target, planting=planting)
        if planting and any(i not in self.harvested and cells[i][1] in ('empty','unknown')
                            and self.crosses_plot(start, target, point)
                            for i, (point, _) in enumerate(mapped)):
            raise PlotActionDeferred('plant_path_crosses_other_plot')
        self.log('plant_drag' if planting else 'harvest_drag', plot=index+1,
                 match_center=center, coordinates=[start,target])
        self.b.drag(*start, *target)
        return menu

    def close_tool(self, frame):
        """Dismiss a known tool only over unchanged grass outside the farm and HUD."""
        dx, dy = self.camera(self.reference, frame)
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.zones[0]
        for y in range(round(.23*h), round(.7*h), 40):
            for x in range(round(.13*w), round(.86*w), 45):
                rx, ry = round(x-dx), round(y-dy)
                if (x0-130 <= rx <= x1+130 and y0-130 <= ry <= y1+130
                        or not (18 <= rx < w-18 and 18 <= ry < h-18)):
                    continue
                patch = frame[y-16:y+17,x-16:x+17]
                ref = self.reference[ry-16:ry+17,rx-16:rx+17]
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                grass = cv2.inRange(hsv, (32,100,76), (87,255,255))
                if cv2.countNonZero(grass) >= grass.size*.97 and cv2.absdiff(patch,ref).mean() < 12:
                    self.tap((x,y), 'verified_grass_close')
                    after = self.stable()
                    return after, self.scan(after)
        raise EvidenceError('tool_still_open_no_verified_dismiss_point')

    def evidence(self, reason, index, before, after):
        stem = f'plot_{index+1}_{time.time_ns()}'
        cv2.imwrite(str(self.debug/(stem+'_before.png')), before)
        cv2.imwrite(str(self.debug/(stem+'_after.png')), after)
        self.log('plot_unconfirmed', plot=index+1, reason=reason, images=stem)

    def watch_transition(self, index, planting, prior, confirmed, allowed):
        source, desired = ('empty','growing') if planting else ('ripe','empty')
        previous_hits = set()
        for attempt in range(5):
            frame = self.stable()
            try:
                cells = self.scan(frame, allow_tool=True)
            except EvidenceError as error:
                # XP and inventory rewards can momentarily cover the farm HUD.
                # Take another picture without sending input. A persistent mismatch stops.
                if str(error) != 'unconfirmed:farm_screen' or attempt == 4:
                    raise
                previous_hits.clear()
                self.log('farm_hud_recheck', plot=index+1)
                continue
            if not self.v.tool_open(frame):
                hits = {i for i in allowed if prior[i][1] == source and cells[i][1] == desired}
                confirmed.update(hits & previous_hits)
                previous_hits = hits
                obscured = any(prior[i][1] == source and cells[i][1] == 'unknown' for i in allowed)
                if index in confirmed and not obscured:
                    return frame, cells
        if self.v.tool_open(frame):
            return self.close_tool(frame)
        return frame, cells

    def pass_plots(self, frame, cells, pending, planting, limit=None):
        source = 'empty' if planting else 'ripe'
        confirmed = self.planted if planting else self.harvested
        for index in pending:
            deferred = None
            if index in confirmed:
                continue
            if limit is not None and len(confirmed) >= limit:
                break
            if cells[index][1] == 'unknown':
                frame = self.stable()
                cells = self.scan(frame)
            if cells[index][1] != source:
                self.evidence('precondition_'+cells[index][1], index, frame, frame)
                continue
            # Retry only while the original state is still positively identified on the farm.
            for _ in range(2 if not planting else 1):
                previous_confirmed = confirmed.copy()
                try:
                    menu = self.open_tool(index, cells, planting)
                except PlotActionDeferred as error:
                    menu = self.action_frame
                    deferred = str(error)
                    if planting:
                        frame, cells = self.close_tool(menu)
                        break
                    frame, cells = menu, self.scan(menu)
                    if cells[index][1] != source:
                        break
                    continue
                deferred = None
                allowed = pending if planting else range(len(cells))
                frame, cells = self.watch_transition(index, planting, cells, confirmed, allowed)
                for hit in sorted(confirmed-previous_confirmed):
                    self.log('plot_planted' if planting else 'plot_harvested',
                             plot=hit+1, coordinates=cells[hit][0])
                if index in confirmed or cells[index][1] != source:
                    break
            if index not in confirmed:
                self.evidence(deferred or 'transition_'+cells[index][1], index, menu, frame)
        return frame, cells

    def harvest_and_plant(self, limit=None):
        self.harvested, self.planted, self.remaining = set(), set(), set()
        pending = []
        try:
            if limit is not None and limit < 1:
                raise EvidenceError('field limit must be positive')
            self.v.require(('farm_screen', 'harvest_tool', 'seed_menu', 'crop_menu_wheat'))
            with self.b.task():
                self.load_plots()
                frame = self.stable()
                cells = self.scan(frame)
                pending = [i for i, (_, state) in enumerate(cells) if state in ('ripe','unknown')]
                frame, cells = self.pass_plots(frame, cells, pending, False, limit)
                self.pass_plots(frame, cells, sorted(self.harvested), True)
                self.reason = 'complete'
        except WORKFLOW_ERRORS as error:
            self.fail(error)
        self.remaining = set(pending) - self.harvested
        unplanted = self.harvested - self.planted
        if self.reason == 'complete' and (unplanted or (self.remaining and (limit is None or len(self.harvested) < limit))):
            self.reason = 'complete_with_remaining'
        result = dict(harvested=len(self.harvested), planted=len(self.planted),
                      remaining=len(self.remaining), unplanted=len(unplanted),
                      remaining_plots=[i+1 for i in sorted(self.remaining)],
                      unplanted_plots=[i+1 for i in sorted(unplanted)], reason=self.reason)
        self.b.last_reason = self.reason
        self.log('harvest_plant_result', **result)
        return result


class VerifiedShopSeller(Workflow):
    def __init__(self, bot, pos):
        super().__init__(bot)
        self.pos = pos
        self.slot = None
        self.price_template = 'price_max_reached'

    def open_shop(self):
        image = self.stable()
        self.v.point('farm_screen', image)
        candidates = self.v.matches('shop_stand', image)
        self.log('shop_candidates', candidates=candidates, source='fresh_striped_roof_template')
        if not candidates:
            raise EvidenceError('shop_not_found')
        if len(candidates)>1 and candidates[0][2]-candidates[1][2]<.03:
            raise EvidenceError('ambiguous_shop')
        self.log('shop_selected', coordinates=candidates[0][:2], reason='highest_template_score_with_unambiguous_margin')
        self.tap(candidates[0][:2], 'fresh_shop_stand')
        self.v.point('shop_screen', self.stable())
        return True

    def tap_empty_slot(self):
        image = self.stable()
        self.v.point('shop_screen',image)
        slots = self.v.matches('shop_slot_empty',image)
        if not slots:
            # A full shop needs its own positive evidence.
            if self.v.state('shop_full', image):
                raise EvidenceError('shop_full')
            raise EvidenceError('shop_slots_unconfirmed')
        self.slot = slots[0][:2]
        self.tap(self.slot,'shop_slot_empty')
        self.v.point('inventory_screen',self.stable())
        return True

    def select_wheat(self):
        image = self.stable()
        self.v.point('inventory_screen',image)
        wheat = self.v.matches('inventory_wheat',image)
        if not wheat:
            self.v.point('out_of_wheat',image)
            raise EvidenceError('out_of_wheat')
        self.tap(wheat[0][:2],'inventory_wheat')
        self.v.point('wheat_selected',self.stable())
        return True

    def read_number(self, name, image):
        x0, y0, x1, y1 = self.v.bounds(name, image)
        crop = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
        ok, png = cv2.imencode('.png', gray)
        if not ok:
            raise EvidenceError('stock_unconfirmed:' + name)
        self.check()
        try:
            result = subprocess.run(
                [self.ocr, 'stdin', 'stdout', '--psm', '7',
                 '-c', 'tessedit_char_whitelist=0123456789xX', 'tsv'],
                input=png.tobytes(), capture_output=True, check=True, timeout=CONFIG['timeout'])
            rows = [row.split('\t') for row in result.stdout.decode().splitlines()
                    if row.startswith('5\t')]
            if len(rows) == 1 and len(rows[0]) == 12:
                confidence, digits = rows[0][-2:]
                if re.fullmatch(r'(0|[1-9][0-9]{0,3})(?:[xX])?', digits) and float(confidence) >= 90:
                    return int(digits.rstrip('xX'))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise EvidenceError('stock_unconfirmed:' + name) from error
        raise EvidenceError('stock_unconfirmed:' + name)

    def confirmed_number(self, name, state):
        readings = []
        for _ in range(2):
            image = self.stable()
            self.v.point(state, image)
            if name == 'inventory_wheat_count' and not self.v.state('inventory_wheat', image):
                raise EvidenceError('stock_unconfirmed:inventory_wheat')
            readings.append(self.read_number(name, image))
        if readings[0] != readings[1]:
            raise EvidenceError('stock_unconfirmed:' + name)
        return readings[0]

    def select_surplus(self, reserve):
        stock = self.confirmed_number('inventory_wheat_count', 'inventory_screen')
        surplus = stock - reserve
        self.log('wheat_stock', confirmed=stock, reserve=reserve, surplus=max(0, surplus))
        if surplus <= 0:
            raise EvidenceError('no_surplus')
        self.select_wheat()
        quantity = self.confirmed_number('sale_wheat_quantity', 'wheat_selected')
        if not 1 <= quantity <= min(stock, 10):
            raise EvidenceError('stock_unconfirmed:sale_wheat_quantity')
        while quantity > surplus:
            image = self.stable()
            self.v.point('wheat_selected', image)
            self.tap(self.v.point('btn_quantity_minus', image), 'btn_quantity_minus')
            next_quantity = self.confirmed_number('sale_wheat_quantity', 'wheat_selected')
            if next_quantity != quantity-1:
                raise EvidenceError('stock_unconfirmed:sale_wheat_quantity')
            quantity = next_quantity
        return quantity

    def exit_shop(self, from_inventory=False):
        if from_inventory:
            image = self.stable()
            for state in ('wheat_selected', 'inventory_screen'):
                if self.v.state(state, image):
                    self.tap(self.v.point('inventory_close', image), 'inventory_close')
                    self.v.point('shop_screen', self.stable())
                    break
            else:
                self.v.point('shop_screen', image)
        else:
            self.v.point('shop_screen', self.stable())
        image = self.stable()
        self.tap(self.v.point('shop_close', image), 'shop_close')
        self.v.point('farm_screen', self.stable())

    def set_max_price(self):
        image = self.stable()
        self.v.point('wheat_selected',image)
        if not self.v.state(self.price_template,image):
            self.tap(self.v.point('btn_price_max',image),'btn_price_max')
        self.v.point(self.price_template,self.stable())
        return True

    def confirm_sale(self, quantity=None):
        image = self.stable()
        self.v.point('wheat_selected',image)
        self.v.point(self.price_template,image)
        if quantity is not None and self.read_number('sale_wheat_quantity', image) != quantity:
            raise EvidenceError('stock_unconfirmed:quantity_changed_before_submission')
        self.tap(self.v.point('btn_put_sale',image),'btn_put_sale')
        # Never repeat an uncertain submission. Reinspect the exact original slot.
        for _ in range(3):
            image = self.stable()
            if self.v.state('shop_screen',image):
                h,w = image.shape[:2]
                x,y = self.slot
                roi = (max(0,(x-60)/w),max(0,(y-55)/h),min(1,(x+60)/w),min(1,(y+55)/h))
                if self.v.matches('shop_slot_wheat', image, roi):
                    self.log('sale_confirmed', coordinates=self.slot, source='same_slot_wheat', before='empty', after='listed_wheat')
                    return True
        raise EvidenceError('sale_unconfirmed')

    def run(self, slots=1, reserve=None, deadline=None):
        sold = 0
        self.price_template = 'price_max_reached'
        try:
            self.reason = 'ready'
            self.v.require(SHOP_TEMPLATES)
            if reserve is not None:
                import shutil
                self.ocr = shutil.which(CONFIG.get('tesseract', 'tesseract'))
                if not self.ocr or any(name not in CONFIG['rois'] for name in
                                       ('inventory_wheat_count', 'sale_wheat_quantity')):
                    self.reason = 'stock_unconfirmed: configure Tesseract and both number ROIs'
                    self.log('sell_skipped', reason=self.reason)
                    self.b.last_reason = self.reason
                    return 0
                self.v.require(('btn_quantity_minus', 'price_max_reached_any_quantity'))
            if slots < 1:
                raise EvidenceError('slots must be positive')
            with self.b.task():
                self.open_shop()
                for _ in range(slots):
                    self.check()
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    try:
                        self.tap_empty_slot()
                    except EvidenceError as error:
                        reason = str(error)
                        if reason != 'shop_full' and not (
                                reason == 'shop_slots_unconfirmed' or
                                reason.startswith('missing_template: shop_full.png')):
                            raise
                        self.exit_shop()
                        self.reason = reason
                        self.log('sell_stopped', reason=reason)
                        break
                    try:
                        if reserve is None:
                            self.select_wheat()
                            quantity = None
                        else:
                            quantity = self.select_surplus(reserve)
                            self.price_template = ('price_max_reached' if quantity == 10
                                                   else 'price_max_reached_any_quantity')
                            self.v.require((self.price_template,))
                        self.set_max_price()
                        self.confirm_sale(quantity)
                    except EvidenceError as error:
                        reason = str(error)
                        if not (reason in ('no_surplus', 'out_of_wheat')
                                or reason.startswith(('stock_unconfirmed:', 'missing_template:'))):
                            raise
                        self.fail(error)
                        self.exit_shop(from_inventory=True)
                        self.log('sell_skipped', reason=reason)
                        break
                    sold += 1
                else:
                    self.reason = 'complete'
                if self.reason in ('ready', 'complete'):
                    self.exit_shop()
                    self.reason = 'complete'
        except WORKFLOW_ERRORS as error:
            self.fail(error)
        self.b.last_reason = self.reason
        self.log('sell_result', slots=sold, reason=self.reason)
        return sold
