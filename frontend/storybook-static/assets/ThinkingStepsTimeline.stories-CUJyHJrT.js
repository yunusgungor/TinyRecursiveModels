import{T as $}from"./ThinkingStepsTimeline-K6WcFrwl.js";import"./jsx-runtime-Bywvkw1S.js";import"./index-CleY8y_P.js";import"./_commonjsHelpers-Cpj98o6Y.js";import"./cn-CytzSlOG.js";const G={title:"Components/ThinkingStepsTimeline",component:$,parameters:{layout:"padded"},tags:["autodocs"],argTypes:{steps:{description:"Array of thinking steps to display in chronological order"},onStepClick:{description:"Callback function when a step is clicked",action:"step clicked"},className:{description:"Additional CSS classes"}}},e=[{step:1,action:"Kullanıcı profilini analiz et",result:"Hobiler ve ilgi alanları belirlendi",insight:"Kullanıcı açık hava aktivitelerini ve sporu tercih ediyor"},{step:2,action:"Kategori filtreleme uygula",result:"5 uygun kategori seçildi",insight:"Spor ve açık hava kategorileri kullanıcı profiliyle eşleşiyor"},{step:3,action:"Bütçe optimizasyonu yap",result:"Fiyat aralığı 500-1500 TL olarak belirlendi",insight:"Kullanıcının bütçesi orta-üst segment ürünlere uygun"},{step:4,action:"Tool sonuçlarını birleştir",result:"Yorum analizi ve trend verileri entegre edildi",insight:"Yüksek puanlı ve trend olan ürünler önceliklendirildi"},{step:5,action:"Final sıralama ve öneriler",result:"10 hediye önerisi oluşturuldu",insight:"Öneriler güven skoruna göre sıralandı"}],U=[...e,{step:6,action:"Stok kontrolü yap",result:"Tüm öneriler stokta mevcut",insight:"Teslimat süresi 2-3 gün"},{step:7,action:"Fiyat karşılaştırması",result:"Rekabetçi fiyatlar belirlendi",insight:"Önerilen ürünler piyasa ortalamasının altında"},{step:8,action:"Kullanıcı geçmişi analizi",result:"Önceki satın alımlar incelendi",insight:"Kullanıcı kaliteli ürünleri tercih ediyor"},{step:9,action:"Sezon uygunluğu kontrolü",result:"Mevsimsel ürünler filtrelendi",insight:"Kış sporları ekipmanları öne çıkarıldı"},{step:10,action:"Final doğrulama",result:"Tüm kriterler karşılandı",insight:"Öneriler kullanıcıya sunulmaya hazır"}],W=[{step:1,action:"Hızlı öneri oluştur",result:"Basit filtreleme uygulandı",insight:"Temel kriterler kullanıldı"}],a={args:{steps:e}},r={args:{steps:W}},n={args:{steps:U}},i={args:{steps:[]}},t={args:{steps:e,onStepClick:c=>{console.log("Clicked step:",c),alert(`Adım ${c.step} tıklandı: ${c.action}`)}}},s={args:{steps:e},parameters:{docs:{description:{story:"Tab tuşu ile adımlar arasında gezinin, Enter veya Space ile detayları açın/kapatın."}}}},l={args:{steps:[{step:3,action:"Üçüncü adım",result:"Sonuç 3",insight:"İçgörü 3"},{step:1,action:"Birinci adım",result:"Sonuç 1",insight:"İçgörü 1"},{step:2,action:"İkinci adım",result:"Sonuç 2",insight:"İçgörü 2"}]},parameters:{docs:{description:{story:"Adımlar otomatik olarak kronolojik sıraya göre düzenlenir."}}}},o={args:{steps:[{step:1,action:"Çok uzun bir aksiyon açıklaması ile detaylı analiz süreci başlatılıyor ve tüm parametreler kontrol ediliyor",result:"Uzun bir sonuç metni: Kullanıcı profili detaylı olarak incelendi, tüm hobiler, ilgi alanları, yaş grubu, bütçe kısıtlamaları ve özel tercihler dikkate alındı. Sistem 150 farklı kategoriyi taradı ve en uygun 5 tanesini belirledi.",insight:"Detaylı içgörü: Kullanıcının geçmiş satın alma davranışları, favori kategorileri, fiyat hassasiyeti ve marka tercihleri analiz edildiğinde, açık hava sporları ve teknoloji ürünlerine yönelik güçlü bir eğilim gözlemlendi. Bu bilgiler ışığında öneriler optimize edildi."},{step:2,action:"İkinci adım",result:"Normal uzunlukta sonuç",insight:"Standart içgörü"}]}},u={args:{steps:e,className:"max-w-2xl mx-auto shadow-xl"}};var p,d,m;a.parameters={...a.parameters,docs:{...(p=a.parameters)==null?void 0:p.docs,source:{originalSource:`{
  args: {
    steps: sampleSteps
  }
}`,...(m=(d=a.parameters)==null?void 0:d.docs)==null?void 0:m.source}}};var g,k,y;r.parameters={...r.parameters,docs:{...(g=r.parameters)==null?void 0:g.docs,source:{originalSource:`{
  args: {
    steps: singleStep
  }
}`,...(y=(k=r.parameters)==null?void 0:k.docs)==null?void 0:y.source}}};var x,S,h;n.parameters={...n.parameters,docs:{...(x=n.parameters)==null?void 0:x.docs,source:{originalSource:`{
  args: {
    steps: longSteps
  }
}`,...(h=(S=n.parameters)==null?void 0:S.docs)==null?void 0:h.source}}};var F,v,C;i.parameters={...i.parameters,docs:{...(F=i.parameters)==null?void 0:F.docs,source:{originalSource:`{
  args: {
    steps: []
  }
}`,...(C=(v=i.parameters)==null?void 0:v.docs)==null?void 0:C.source}}};var b,z,f;t.parameters={...t.parameters,docs:{...(b=t.parameters)==null?void 0:b.docs,source:{originalSource:`{
  args: {
    steps: sampleSteps,
    onStepClick: step => {
      console.log('Clicked step:', step);
      alert(\`Adım \${step.step} tıklandı: \${step.action}\`);
    }
  }
}`,...(f=(z=t.parameters)==null?void 0:z.docs)==null?void 0:f.source}}};var E,T,K;s.parameters={...s.parameters,docs:{...(E=s.parameters)==null?void 0:E.docs,source:{originalSource:`{
  args: {
    steps: sampleSteps
  },
  parameters: {
    docs: {
      description: {
        story: 'Tab tuşu ile adımlar arasında gezinin, Enter veya Space ile detayları açın/kapatın.'
      }
    }
  }
}`,...(K=(T=s.parameters)==null?void 0:T.docs)==null?void 0:K.source}}};var N,O,A;l.parameters={...l.parameters,docs:{...(N=l.parameters)==null?void 0:N.docs,source:{originalSource:`{
  args: {
    steps: [{
      step: 3,
      action: 'Üçüncü adım',
      result: 'Sonuç 3',
      insight: 'İçgörü 3'
    }, {
      step: 1,
      action: 'Birinci adım',
      result: 'Sonuç 1',
      insight: 'İçgörü 1'
    }, {
      step: 2,
      action: 'İkinci adım',
      result: 'Sonuç 2',
      insight: 'İçgörü 2'
    }]
  },
  parameters: {
    docs: {
      description: {
        story: 'Adımlar otomatik olarak kronolojik sıraya göre düzenlenir.'
      }
    }
  }
}`,...(A=(O=l.parameters)==null?void 0:O.docs)==null?void 0:A.source}}};var B,w,D;o.parameters={...o.parameters,docs:{...(B=o.parameters)==null?void 0:B.docs,source:{originalSource:`{
  args: {
    steps: [{
      step: 1,
      action: 'Çok uzun bir aksiyon açıklaması ile detaylı analiz süreci başlatılıyor ve tüm parametreler kontrol ediliyor',
      result: 'Uzun bir sonuç metni: Kullanıcı profili detaylı olarak incelendi, tüm hobiler, ilgi alanları, yaş grubu, bütçe kısıtlamaları ve özel tercihler dikkate alındı. Sistem 150 farklı kategoriyi taradı ve en uygun 5 tanesini belirledi.',
      insight: 'Detaylı içgörü: Kullanıcının geçmiş satın alma davranışları, favori kategorileri, fiyat hassasiyeti ve marka tercihleri analiz edildiğinde, açık hava sporları ve teknoloji ürünlerine yönelik güçlü bir eğilim gözlemlendi. Bu bilgiler ışığında öneriler optimize edildi.'
    }, {
      step: 2,
      action: 'İkinci adım',
      result: 'Normal uzunlukta sonuç',
      insight: 'Standart içgörü'
    }]
  }
}`,...(D=(w=o.parameters)==null?void 0:w.docs)==null?void 0:D.source}}};var L,j,H;u.parameters={...u.parameters,docs:{...(L=u.parameters)==null?void 0:L.docs,source:{originalSource:`{
  args: {
    steps: sampleSteps,
    className: 'max-w-2xl mx-auto shadow-xl'
  }
}`,...(H=(j=u.parameters)==null?void 0:j.docs)==null?void 0:H.source}}};const I=["Default","SingleStep","LongTimeline","EmptySteps","WithClickHandler","KeyboardNavigation","OutOfOrderSteps","LongTextContent","CustomStyling"];export{u as CustomStyling,a as Default,i as EmptySteps,s as KeyboardNavigation,o as LongTextContent,n as LongTimeline,l as OutOfOrderSteps,r as SingleStep,t as WithClickHandler,I as __namedExportsOrder,G as default};
