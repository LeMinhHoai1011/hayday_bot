"""Offline regression checks; never connect to or send input to BlueStacks."""
import contextlib
import tempfile
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from verified_workflow import Vision, EvidenceError, VerifiedShopSeller, FarmWorkflow, Stopped
from bot import BlueStacksBot

class Bot:
    device = 'test'
    def __init__(self):
        self.stop_event = threading.Event()
        self.taps = []
        self.phase = 'farm_screen'
        self.slots = 0
        self.last_reason = ''
        self.fail_sale = False
        self.full = False
        self.no_wheat = False
    def check_stop(self):
        if self.stop_event.is_set():
            raise Stopped('stopped')
    @contextlib.contextmanager
    def task(self):
        self.check_stop()
        yield
    def screenshot(self):
        return np.zeros((200,300,3), np.uint8)
    def tap(self,x,y):
        self.check_stop()
        self.taps.append((x,y))
        self.phase = {(10,10):'shop_screen',(50,50):'inventory_screen',(150,50):'inventory_screen',(20,20):'wheat_selected',(30,30):'max',(40,40):'shop_screen'}[(x,y)]
        if (x,y)==(40,40) and not self.fail_sale:
            self.slots+=1
    def back(self):
        self.check_stop()
        self.phase='farm_screen'

class SemanticVision:
    def __init__(self,b): self.b=b
    def matches(self,name,image,roi=None):
        b=self.b
        if name=='shop_stand': return [(10,10,.99)]
        if name=='shop_slot_empty': return [] if b.full else [(50+100*b.slots,50,.99)]
        if name=='shop_full': return [(1,1,.99)] if b.full else []
        if name=='inventory_wheat': return [] if b.no_wheat else [(20,20,.99)]
        if name=='out_of_wheat': return [(1,1,.99)] if b.no_wheat else []
        if name=='btn_price_max': return [(30,30,.99)]
        if name=='btn_put_sale': return [(40,40,.99)]
        if name=='shop_slot_wheat':
            if b.slots and roi is not None:
                # Require the requested region to contain the slot just submitted.
                expected=50+100*(b.slots-1)
                return [(expected,50,.99)] if roi[0]*300<=expected<=roi[2]*300 else []
            return []
        if name=='wheat_selected': return [(1,1,.99)] if b.phase in ('wheat_selected','max') else []
        if name=='price_max_reached': return [(1,1,.99)] if b.phase=='max' else []
        return [(1,1,.99)] if name==b.phase else []
    def point(self,name,image,roi=None):
        matches=self.matches(name,image,roi)
        if not matches: raise EvidenceError('unconfirmed:'+name)
        return matches[0][:2]
    def state(self,name,image): return bool(self.matches(name,image))

class Regression(unittest.TestCase):
    def seller(self,b):
        s=VerifiedShopSeller(b,None)
        s.v=SemanticVision(b)
        s.stable=b.screenshot
        s.log=lambda *a,**k:None
        s.fail=lambda error,*a:setattr(s,'reason',str(error))
        return s
    def test_two_slots(self):
        b=Bot();s=self.seller(b)
        self.assertEqual(s.run(2),2)
        self.assertEqual(s.reason,'complete')
        self.assertEqual(b.taps.count((40,40)),2)
    def test_unconfirmed_submission_not_repeated(self):
        b=Bot();b.fail_sale=True;s=self.seller(b)
        self.assertEqual(s.run(2),0)
        self.assertEqual(s.reason,'sale_unconfirmed')
        self.assertEqual(b.taps.count((40,40)),1)
    def test_full_shop(self):
        b=Bot();b.full=True;s=self.seller(b)
        self.assertEqual(s.run(2),0)
        self.assertEqual(s.reason,'shop_full')
    def test_no_wheat(self):
        b=Bot();b.no_wheat=True;s=self.seller(b)
        self.assertEqual(s.run(2),0)
        self.assertEqual(s.reason,'out_of_wheat')
    def test_stop_prevents_next_tap(self):
        b=Bot();s=self.seller(b)
        old=s.select_wheat
        def stop():
            old();b.stop_event.set()
        s.select_wheat=stop
        self.assertEqual(s.run(2),0)
        self.assertNotIn((30,30),b.taps)
        self.assertNotIn((40,40),b.taps)
    def test_camera_translation(self):
        rng=np.random.default_rng(42)
        before=rng.integers(0,256,(400,600,3),dtype=np.uint8)
        after=cv2.warpAffine(before,np.float32([[1,0,9],[0,1,-6]]),(600,400))
        dx,dy=Vision(None).camera(before,after,[(200,100,400,300)])
        self.assertAlmostEqual(dx,9,delta=.8)
        self.assertAlmostEqual(dy,-6,delta=.8)
    def test_missing_landmarks(self):
        image=np.zeros((400,600,3),np.uint8)
        with self.assertRaises(EvidenceError): Vision(None).camera(image,image)
    def test_missing_template_stops_without_input(self):
        b=Bot();s=self.seller(b);s.v=Vision(b)
        with tempfile.TemporaryDirectory() as directory, patch.dict('verified_workflow.CONFIG',{'templates_dir':directory}):
            self.assertEqual(s.run(),0)
        self.assertEqual(b.taps,[])
        self.assertEqual(s.reason,'missing_template:farm_screen')
    def test_adb_failure_propagates(self):
        import subprocess
        b=BlueStacksBot()
        with patch('bot.subprocess.run',side_effect=subprocess.CalledProcessError(1,['adb'])):
            with self.assertRaises(subprocess.CalledProcessError): b.tap(1,2)
    def test_camera_drift_stops_user_zones(self):
        b=Bot();f=FarmWorkflow(b);f.stable=b.screenshot
        f.log=lambda *a,**k:None
        f.fail=lambda error,*a:setattr(f,'reason',str(error))
        class V:
            def point(self,*a): return (20,20)
            def matches(self,name,*a): return [(50,50,.99)]
            def camera(self,*a): return (10,0)
        f.v=V()
        self.assertEqual(f.run(zones=[(0,0,100,100)]),0)
        self.assertEqual(f.reason,'camera_moved_opening_tool')

class Integration(unittest.TestCase):
    def test_menu_8_through_13(self):
        import run
        for choice in range(8,14):
            answers = [str(choice)] + (["2"] if choice in (8,11) else []) + ["0"]
            with self.subTest(choice=choice), patch("builtins.input",side_effect=answers), patch("builtins.print"), patch("run.Positions") as positions, patch("run.ShopSeller") as seller, patch("run.UpgradeManager") as upgrade, patch("run.calibrate",return_value="farm.png"), patch("run.calibrate_shop",return_value="shop.png"), patch("subprocess.Popen") as launch:
                bot=object()
                run.menu(bot)
                if choice in (8,11):
                    seller.assert_called_once_with(bot,positions.return_value)
                    seller.return_value.run.assert_called_once_with(slots=2)
                if choice in (9,10,11):
                    upgrade.assert_called_once_with(bot,positions.return_value)
                if choice in (12,13):
                    launch.assert_called_once_with(["explorer","farm.png" if choice==12 else "shop.png"])
    def test_device_lock_excludes_other_thread(self):
        first=BlueStacksBot();second=BlueStacksBot();results=[]
        def concurrent():
            try:
                with second.task(): results.append("entered")
            except RuntimeError: results.append("busy")
        with first.task():
            t=threading.Thread(target=concurrent);t.start();t.join(3)
        self.assertEqual(results,["busy"])
        with second.task(): pass

if __name__=='__main__': unittest.main()
